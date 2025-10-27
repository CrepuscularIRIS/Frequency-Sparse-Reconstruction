
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_dct import dct_2d, idct_2d # 从安装的库中导入函数


class FSR(nn.Module):
    """
    Frequency-domain Sparse Reconstruction (FSR) 上采样层
    - Input: (B, C, H, W)
    - Output: (B, C, H*s, W*s)  —— 默认 s=2
    """

    def __init__(self, c1, scale: int = 2, share_mask: bool = True, init: str = "ones", sparsity_lambda: float = 1e-4):
        """
        Initializes the FSR module.
        Note: `c2` from `parse_model` is repurposed as `scale`. `c1` is `in_channels`.
        """
        super().__init__()
        self.in_channels = c1
        self.scale = scale
        self.sparsity_lambda = sparsity_lambda

        # 可学习稀疏掩码：实数、可微
        mask_shape = (1, 1, 1, 1) if share_mask else (1, self.in_channels, 1, 1)
        mask = torch.ones(mask_shape) if init == "ones" else torch.randn(mask_shape) * 0.01
        self.mask = nn.Parameter(mask)
        self.reg = 0.0

    def forward(self, x):
        """
        Step 1) rFFT2  -->  Step 2) 稀疏高频调制  -->  Step 3) irFFT2 (size ↑)
        为减少峰值显存，(1)+(2) 走 checkpoint，只在需要梯度回传时重新计算。
        """
        B, C, H, W = x.shape
        scale = self.scale

        def _freq_branch(inp):
            fr = torch.fft.rfft2(inp, norm="ortho")  # (B,C,H,W/2+1), complex64/16
            fr = fr * self.mask.to(fr.dtype)  # 稀疏激活
            return fr

        # checkpoint 节省中间激活显存（尤其 batch>1, H,W>128 时）
        fr = checkpoint(_freq_branch, x, use_reentrant=True) if self.training else _freq_branch(x)
        out = torch.fft.irfft2(fr, s=(H * scale, W * scale), norm="ortho")

        # 稀疏 L1 正则（仅训练时）——在外层 loss 汇总
        if self.training and self.sparsity_lambda > 0:
            self.reg = self.sparsity_lambda * torch.mean(torch.abs(self.mask))
        else:
            self.reg = 0.0

        return out


class LWT_FSR(FSR):
    """
    可学习小波 → 稀疏高频 → 逆小波
    相比 DCT，全局频谱 → (时‑频) 局部化，更擅长 IR 场景的非平稳背景抑制
    """

    def __init__(self, c1, scale: int = 2, wave="haar", **kwargs):
        super().__init__(c1, scale=scale, **kwargs)
        if scale != 2:
            raise NotImplementedError("LWT_FSR currently only supports scale_factor=2")
        # Decomposition kernels
        k = torch.tensor([1.0, 1.0]) / 2 if wave == "haar" else torch.tensor([-0.125, 0.25, 0.75, 0.25, -0.125])
        self.register_buffer("lf_kernel", k.view(1, 1, -1, 1))

        hf_k_coeffs = torch.flip(k, [-1])
        sign = torch.ones_like(hf_k_coeffs)
        sign[1::2] = -1
        hf_k_coeffs = hf_k_coeffs * sign
        self.register_buffer("hf_kernel", hf_k_coeffs.view(1, 1, -1, 1))

        # Reconstruction kernels (for Haar, they are the same as decomposition)
        self.register_buffer("lf_inv_kernel", self.lf_kernel)
        self.register_buffer("hf_inv_kernel", self.hf_kernel)

    def _wavelet_fwd(self, x):
        h, w = x.shape[-2:]
        # Pad to match backbone conv stride-2 behavior (ceil division)
        padding_bottom = 1 if h % 2 != 0 else 0
        padding_right = 1 if w % 2 != 0 else 0
        if padding_bottom or padding_right:
            x = F.pad(x, (0, padding_right, 0, padding_bottom))
        
        conv_padding = (self.lf_kernel.shape[2] - 1) // 2
        lf = F.conv2d(x, self.lf_kernel.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1), padding=conv_padding)
        hf = F.conv2d(x, self.hf_kernel.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1), padding=conv_padding)
        return lf, hf

    def _wavelet_inv(self, lf, hf, target_size):
        """
        使用转置卷积实现一个更严谨的逆小波变换, 并用interpolate保证尺寸。
        """
        c = lf.size(1)
        k_size = self.lf_inv_kernel.shape[2]
        padding = (k_size - 1) // 2

        out_lf = F.conv_transpose2d(
            lf, self.lf_inv_kernel.expand(c, -1, -1, -1), stride=self.scale, groups=c, padding=padding
        )
        out_hf = F.conv_transpose2d(
            hf, self.hf_inv_kernel.expand(c, -1, -1, -1), stride=self.scale, groups=c, padding=padding
        )
        reconstructed = out_lf + out_hf

        # Use interpolate as the final step to guarantee the exact output size
        # This is more robust than cropping or padding.
        if reconstructed.shape[-2:] != target_size:
            reconstructed = F.interpolate(reconstructed, size=target_size, mode="bilinear", align_corners=False)

        return reconstructed

    def forward(self, x):
        target_size = (x.shape[2] * self.scale, x.shape[3] * self.scale)
        # Store original shape for cropping in inverse transform.
        self.original_size = x.shape[-2:]
        lf, hf = self._wavelet_fwd(x)
        # 对 HF 子带做稀疏门控（共享 mask、可加 soft‑shrink）
        hf = hf * self.mask.to(hf.dtype)
        out = self._wavelet_inv(lf, hf, target_size)
        return out


class FourierUpLite_FSR(FSR):
    """
    仅调制 amplitude spectrum，phase copy 并在 irfft2 时注入。
    适合低 SNR、目标边缘模糊场景（幅值峰值是主判别）。
    """
    @staticmethod
    def _freq_pad(xr, scale):
        """Helper function to pad frequency domain tensors for upsampling."""
        B, C, H, W2 = xr.shape
        pad_H = H * (scale - 1)
        W_new = (W2 - 1) * scale + 1
        pad_W2 = W_new - W2
        return F.pad(xr, (0, pad_W2, 0, pad_H))

    def forward(self, x):
        B, C, H, W = x.shape
        fr = torch.fft.rfft2(x, norm="ortho")  # complex
        amp, phase = torch.abs(fr), torch.angle(fr)
        amp = amp * self.mask.to(amp.dtype)      # 调幅
        # 零填+复数重组
        amp_padded = self._freq_pad(amp, self.scale)
        phase_padded = self._freq_pad(phase, self.scale)
        fr_padded = torch.polar(amp_padded, phase_padded)
        out = torch.fft.irfft2(fr_padded, s=(H * self.scale, W * self.scale), norm="ortho")
        return out


class FSR_FixedHPF(nn.Module):
    """
    FSR with a fixed High-Pass Filter, simulating HS-FPN.
    The high-pass filter is created in the frequency domain and is not learnable.
    This version is rewritten to be simpler and more correct.
    """

    def __init__(self, c1, scale: int = 2, hpf_radius_ratio=0.2, *args):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(c1, c1, kernel_size=1)

        # Create a fixed high-pass filter mask template.
        # It's a small template that will be resized on the fly.
        self.register_buffer('h_mask', self.create_fixed_hpf_mask(size=32, radius_ratio=hpf_radius_ratio))

    @staticmethod
    def create_fixed_hpf_mask(size=32, radius_ratio=0.2):
        """
        Creates a fixed high-pass filter mask directly in the rfft frequency domain.
        It's a circle centered at the DC component (top-left corner).
        """
        h = w = size
        w_rfft = w // 2 + 1
        
        # Create a grid of coordinates
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w_rfft), indexing='ij')

        # Define the radius of the low-frequency circle to cut out
        radius = int(min(h, w) * radius_ratio)
        
        # The mask is 1 where distance from DC (0,0) is greater than radius
        dist_from_dc = torch.sqrt(Y**2 + X**2)
        hpf_mask = (dist_from_dc > radius).float()
        
        # Return a 4D mask (1, 1, H, W_rfft) for broadcasting
        return hpf_mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape

        # 1. Transform to frequency domain
        x_freq = torch.fft.rfft2(x, norm="ortho")

        # 2. Resize the HPF mask to match the feature map's frequency domain size
        h_mask_resized = F.interpolate(self.h_mask, size=x_freq.shape[-2:], mode='bilinear', align_corners=False)
        
        # 3. Apply the high-pass filter. The mask will broadcast across batch and channel dimensions.
        x_freq_filtered = x_freq * h_mask_resized

        # 4. Inverse FFT to get the upscaled image.
        #    irfft2 handles the zero-padding internally when a larger size `s` is given.
        x_upscaled = torch.fft.irfft2(x_freq_filtered, s=(H * self.scale, W * self.scale), norm="ortho")

        return x_upscaled


class DCT_FSR(nn.Module):
    """
    基于离散余弦变换（DCT）的频域稀疏重建模块。
    该模块的设计严格遵循提案文档中的伪代码和分析报告中的理论。
    """
    def __init__(self, c1, scale: int = 2, *args):
        super().__init__()
        self.scale_factor = scale
        
        # 定义可学习的稀疏高频滤波器 (对应伪代码中的 sparse_mask)
        # 这是一个可学习的网络参数，通过端到端的训练动态优化
        self.sparse_mask = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x_lowres):
        """
        前向传播逻辑
        输入: x_lowres (低分辨率特征图)
        输出: 重建后的高分辨率特征图
        """
        B, C, H, W = x_lowres.shape
        target_size = (H * self.scale_factor, W * self.scale_factor)

        # 1. 空间域 -> DCT域
        #    使用 torch_dct 库提供的 dct_2d 函数
        dct_coeffs = dct_2d(x_lowres, norm='ortho')

        # 2. 在DCT系数域通过补零（Zero-Padding）实现上采样
        dct_coeffs_padded = F.pad(dct_coeffs, (0, W * (self.scale_factor - 1), 0, H * (self.scale_factor - 1)))
        
        # 3. 应用可学习掩码进行高频稀疏激活
        #    将掩码插值到目标尺寸以匹配补零后的系数矩阵
        mask_resized = F.interpolate(self.sparse_mask, size=target_size, mode='bilinear', align_corners=False)
        dct_coeffs_modulated = dct_coeffs_padded * mask_resized
        
        # 4. DCT域 -> 重建空间特征
        #    使用 torch_dct 库提供的 idct_2d 函数
        reconstructed_features = idct_2d(dct_coeffs_modulated, norm='ortho')

        return reconstructed_features

