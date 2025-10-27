import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 1. 基础：2‑D DCT / iDCT ---------------------------------------------------
try:
    from torch_dct import dct_2d, idct_2d
except ImportError as e:
    raise ImportError("缺少 torch‑dct，请先安装： pip install torch-dct") from e


def radius_grid(h: int, w: int, device):
    """归一化到 0~0.5 的半径网格 r(u,v)。"""
    y = torch.linspace(-0.5, 0.5, h, device=device).unsqueeze(1)
    x = torch.linspace(-0.5, 0.5, w, device=device).unsqueeze(0)
    return torch.sqrt(x ** 2 + y ** 2)  # H×W


# --------------------------------------------------------------------------
# 2. 核心算子：FSR & SCR ----------------------------------------------------
class LearnableMask(nn.Module):
    """用 (t,k) 两个标量参数化的圆环状软掩码，既可解释又省显存。"""
    def __init__(self, init_t=0.25, init_k=3.0):
        super().__init__()
        # Use logit to avoid direct optimization on bounded values
        self.logit_t = nn.Parameter(torch.tensor(math.log(init_t / (0.5 - init_t))))
        self.logit_k = nn.Parameter(torch.tensor(math.log(init_k / 10)))

    def forward(self, h, w, device):
        r = radius_grid(h, w, device)
        t = torch.sigmoid(self.logit_t) * 0.5            # (0,0.5)
        k = torch.sigmoid(self.logit_k) * 10 + 1e-3      # (1e‑3,10)
        return torch.sigmoid((r - t) * k)                # 高频 ≈ 1，低频 ≈ 0


class FSR(nn.Module):
    """
    频域稀疏重建：
        输入  : 深层低分辨 F_H  +  高频引导 G
        输出  : F_SR (分辨率 = F_H)
        机制  : 用可学习掩码，将高频从 G 注入 F_H
    """
    def __init__(self, ch_deep, ch_guide, ch_mid):
        super().__init__()
        self.align_H = nn.Conv2d(ch_deep,  ch_mid, 1, bias=False)
        self.align_G = nn.Conv2d(ch_guide, ch_mid, 1, bias=False)
        self.mask    = LearnableMask(init_t=0.35, init_k=4.0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, F_H, G):
        H = self.align_H(F_H)           # N,C,H,W
        G = self.align_G(G)
        S_H, S_G = dct_2d(H), dct_2d(G)

        M = self.mask(S_H.size(-2), S_H.size(-1), S_H.device).unsqueeze(0).unsqueeze(0)
        # 注入：低频保留 H，高频来自 G
        S_SR = (1 - M) * S_H + M * S_G
        F_SR = self.act(idct_2d(S_SR))
        return F_SR


class SCR(nn.Module):
    """
    频谱‑拼接重构：与前一回复中的实现一致，但独立出来便于级联。
    """
    def __init__(self, ch_low, ch_high, ch_mid):
        super().__init__()
        self.align_L  = nn.Conv2d(ch_low,  ch_mid, 1, bias=False)
        self.align_H  = nn.Conv2d(ch_high, ch_mid, 1, bias=False)
        self.mask     = LearnableMask(init_t=0.20, init_k=3.5)
        self.fuse     = nn.Conv2d(ch_mid * 3, ch_mid, 1, bias=False)
        self.act      = nn.ReLU(inplace=True)

    def forward(self, F_L, F_H):
        L = self.align_L(F_L)
        H = self.align_H(F_H)
        S_L, S_H = dct_2d(L), dct_2d(H)

        M = self.mask(S_L.size(-2), S_L.size(-1), S_L.device).unsqueeze(0).unsqueeze(0)
        S_F = M * S_L + (1 - M) * S_H
        F_F = idct_2d(S_F)

        x = torch.cat([F_F, L, H], dim=1)
        return self.act(self.fuse(x))


# --------------------------------------------------------------------------
# 3. 通用卷积块（与原 Unet 相同） -------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.pool_conv(x)


# --------------------------------------------------------------------------
# 4. UpPMR：FSR → SCR → Conv -----------------------------------------------
class UpPMR(nn.Module):
    def __init__(self, ch_deep, ch_skip, ch_out, bilinear=True):
        super().__init__()
        self.up = (
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if bilinear else nn.ConvTranspose2d(ch_deep, ch_deep, 2, stride=2)
        )
        ch_mid = ch_out
        self.gate = nn.Sequential(
            nn.Conv2d(ch_skip + ch_deep, ch_mid, 1, bias=False),
            nn.Sigmoid()
        )
        self.fsr  = FSR(ch_deep, ch_mid, ch_mid)
        self.scr  = SCR(ch_skip, ch_mid, ch_mid)
        self.conv = DoubleConv(ch_mid, ch_out)

    def forward(self, x_deep, x_skip):
        x_deep_up = self.up(x_deep)
        dy, dx = x_skip.size(2) - x_deep_up.size(2), x_skip.size(3) - x_deep_up.size(3)
        if dy or dx:
            x_deep_up = F.pad(x_deep_up, [dx // 2, dx - dx//2, dy // 2, dy - dy//2])

        guide = self.gate(torch.cat([x_skip, x_deep_up], dim=1)) * x_skip
        x_sr = self.fsr(x_deep_up, guide)
        x_fuse = self.scr(x_skip, x_sr)
        return self.conv(x_fuse)


# --------------------------------------------------------------------------
# 5. DeclarativePMRNet v2 主体 -----------------------------------------------
class DeclarativePMRNet_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_c=32, levels=4, Ce=32, bilinear=True):
        super().__init__()
        assert levels in {3, 4, 5}, "目前仅支持 3‑5 级解码"
        self.Ce = Ce
        c = [base_c * k for k in [1, 2, 4, 8, 16, 32]]

        # Encoder
        self.inc   = DoubleConv(n_channels, c[0])
        self.down1 = Down(c[0],  c[1])
        self.down2 = Down(c[1],  c[2])
        self.down3 = Down(c[2],  c[3])
        self.down4 = Down(c[3],  c[4]) if levels >= 4 else None
        self.down5 = Down(c[4],  c[5]) if levels == 5 else None

        # Decoder (PMR blocks)
        self.decoder_levels = levels
        if levels == 5:
            self.up1 = UpPMR(c[5], c[4], c[4], bilinear)
            self.up2 = UpPMR(c[4], c[3], c[3], bilinear)
            self.up3 = UpPMR(c[3], c[2], c[2], bilinear)
            self.up4 = UpPMR(c[2], c[1], c[1], bilinear)
            self.up5 = UpPMR(c[1], c[0], c[0], bilinear)
        elif levels == 4:
            self.up1 = UpPMR(c[4], c[3], c[3], bilinear)
            self.up2 = UpPMR(c[3], c[2], c[2], bilinear)
            self.up3 = UpPMR(c[2], c[1], c[1], bilinear)
            self.up4 = UpPMR(c[1], c[0], c[0], bilinear)
        else:  # levels == 3
            self.up1 = UpPMR(c[3], c[2], c[2], bilinear)
            self.up2 = UpPMR(c[2], c[1], c[1], bilinear)
            self.up3 = UpPMR(c[1], c[0], c[0], bilinear)

        # Declarative Head
        self.embedding_head = nn.Conv2d(c[0], self.Ce, kernel_size=1, bias=False)
        self.outc = nn.Conv2d(self.Ce, n_classes, kernel_size=1)
        self.raw_g = nn.Parameter(torch.zeros(self.Ce), requires_grad=True)

        # Multi-scale Supervision Heads
        self.ms_head3 = nn.Conv2d(c[2], n_classes, 1)
        self.ms_head2 = nn.Conv2d(c[1], n_classes, 1)
        self.ms_head1 = nn.Conv2d(c[0], n_classes, 1)


    def forward(self, x):
        g = torch.tanh(self.raw_g) * 3
        side_outputs = []

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.down4 is not None:
            x5 = self.down4(x4)
        if self.down5 is not None:
            x6 = self.down5(x5)

        # Decoder
        if self.decoder_levels == 5:
            d = self.up1(x6, x5)
            d = self.up2(d,  x4)
            side_outputs.append(self.ms_head3(d))
            d = self.up3(d,  x3)
            side_outputs.append(self.ms_head2(d))
            d = self.up4(d,  x2)
            side_outputs.append(self.ms_head1(d))
            d = self.up5(d,  x1)
        elif self.decoder_levels == 4:
            d = self.up1(x5, x4)
            d = self.up2(d,  x3)
            side_outputs.append(self.ms_head3(d))
            d = self.up3(d,  x2)
            side_outputs.append(self.ms_head2(d))
            d = self.up4(d,  x1)
            side_outputs.append(self.ms_head1(d))
        else: # 3 levels
            d = self.up1(x4, x3)
            side_outputs.append(self.ms_head3(d))
            d = self.up2(d,  x2)
            side_outputs.append(self.ms_head2(d))
            d = self.up3(d,  x1)
            side_outputs.append(self.ms_head1(d))

        embedding_map = self.embedding_head(d)
        logits = self.outc(embedding_map)

        if self.training:
            return (embedding_map, logits, g, self.raw_g), side_outputs[::-1] # Reverse for largest scale first
        else:
            return embedding_map, logits, g, self.raw_g

    def freeze_arm(self):
        self.raw_g.requires_grad = False
    def unfreeze_arm(self):
        self.raw_g.requires_grad = True 