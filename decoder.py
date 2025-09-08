# ============================================================
# ART Multi‑Layer Transparency Decoder (PyTorch)
# Implements: ViT‑based multi‑layer transparency decoder with 3D‑RoPE,
# layout‑guided paste/unpatchify, and a minimal training wrapper.
# Compatible with Flux VAE encoder (frozen) from diffusers AutoencoderKL.
#
# Paper references (for your convenience while coding):
#  - EVAE: downsample x8, 16‑ch latent (ART Fig.4 & Sec.3.1)
#  - Decoder: Linearin 16->768, ViT‑B, Linearout 768->256 -> 8x8x4 RGBA patches
#  - Ceiling‑aligned tight crop: H,W divisible by 16
#  - 3D‑RoPE over (x, y, layer)
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utilities: patchify/unpatchify
# ---------------------------

def unpatchify_rgba(patch_tokens: torch.Tensor, h: int, w: int, patch: int = 8) -> torch.Tensor:
    """
    Convert tokens of shape [B, (h/patch)*(w/patch), patch*patch*4] -> [B, 4, h, w]
    Assumes row‑major token ordering.
    """
    B, N, C = patch_tokens.shape
    assert C == patch * patch * 4, f"Expect last dim={patch*patch*4}, got {C}"
    gh, gw = h // patch, w // patch
    assert N == gh * gw, f"Tokens {N} != grid {gh}x{gw}"
    x = patch_tokens.view(B, gh, gw, 4, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 4, h, w)
    return x


def patchify_latent(latent: torch.Tensor) -> torch.Tensor:
    """
    EVAE latent -> tokens
    latent: [B, 16, h8, w8]  (16 channels, downsample /8)
    returns tokens z: [B, N, 16] with N = h8*w8 in row‑major order.
    """
    B, C, h8, w8 = latent.shape
    assert C == 16, f"Flux EVAE latent should have 16 channels, got {C}"
    z = latent.permute(0, 2, 3, 1).contiguous().view(B, h8 * w8, C)
    return z


# ---------------------------
# Layout spec
# ---------------------------

@dataclass
class SegmentSpec:
    name: str                    # e.g., 'merged', 'background', 'fg_000'
    kind: str                    # {'merged','background','foreground'}
    h: int                       # pixel height of the segment region
    w: int                       # pixel width of the segment region
    layer_id: int                # layer index used in 3D‑RoPE
    bbox_xyxy: Optional[Tuple[int,int,int,int]] = None  # only for fg (x0,y0,x1,y1) in the full canvas

@dataclass
class LayoutSpec:
    segments: List[SegmentSpec]

    def token_shapes(self, patch: int = 8) -> Dict[str, Tuple[int, int]]:
        out = {}
        for seg in self.segments:
            assert seg.h % patch == 0 and seg.w % patch == 0, \
                f"{seg.name} size must be divisible by patch={patch}"
            out[seg.name] = (seg.h // patch, seg.w // patch)
        return out

    def concat_positions_abs(self, device: torch.device, patch: int = 8) -> torch.Tensor:
        """
        构造 (layer, y, x) 的**绝对**坐标 ids，shape [sum_N, 3]
        - merged/background：y,x 为全图坐标
        - foreground：y,x = bbox 原点(//patch) + 局部网格
        """
        pos_list = []
        for seg in self.segments:
            h8, w8 = seg.h // patch, seg.w // patch
            yy, xx = torch.meshgrid(
                torch.arange(h8, device=device),
                torch.arange(w8, device=device),
                indexing='ij'
            )
            if seg.kind == 'foreground' and seg.bbox_xyxy is not None:
                x0, y0, _, _ = seg.bbox_xyxy
                xx = xx + (x0 // patch)
                yy = yy + (y0 // patch)
            ll = torch.full_like(xx, fill_value=seg.layer_id)        # layer 绝对 id
            ids = torch.stack([ll, yy, xx], dim=-1).view(-1, 3)      # (layer, y, x)
            pos_list.append(ids)
        return torch.cat(pos_list, dim=0)

    def segment_token_slices(self, patch: int = 8) -> Dict[str, slice]:
        slices = {}
        start = 0
        for seg in self.segments:
            h8, w8 = seg.h // patch, seg.w // patch
            n = h8 * w8
            slices[seg.name] = slice(start, start + n)
            start += n
        return slices

# ---------------------------
# 3D RoPE utilities
# ---------------------------

class RoPE3D(nn.Module):
    """
    3D Rotary Positional Embedding over (layer, y, x).
    - head_dim = dl + dy + dx
    - 默认 dl=8（不超过 head_dim/4），余下均分给 y/x
    - 与官方实现一致：只对 Q/K 施加 RoPE
    """
    def __init__(self, head_dim: int, theta: float = 10000.0, dims: Tuple[int,int,int] = None):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        if dims is None:
            dl = min(8, head_dim // 4)
            rem = head_dim - dl
            dy = rem // 2
            dx = rem - dy
            dims = (dl, dy, dx)          # (layer, y, x)
        dl, dy, dx = dims
        assert dl + dy + dx == head_dim, f"RoPE dims must sum to head_dim, got {(dl,dy,dx)} vs {head_dim}"
        self.dl, self.dy, self.dx = dl, dy, dx

    @staticmethod
    def _freqs(dim: int, pos: torch.Tensor, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # pos: [S], returns cos/sin: [S, dim]
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device) / dim))
        ang = pos[:, None] * inv_freq[None, :]
        cos = torch.repeat_interleave(ang.cos(), repeats=2, dim=-1)
        sin = torch.repeat_interleave(ang.sin(), repeats=2, dim=-1)
        return cos, sin

    def get_cos_sin(self, ids_lyx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ids_lyx: [S, 3] with (layer, y, x)
        返回拼接后的 cos/sin: [S, head_dim]
        """
        l = ids_lyx[:, 0]
        y = ids_lyx[:, 1]
        x = ids_lyx[:, 2]
        cos_l, sin_l = self._freqs(self.dl, l, self.theta)
        cos_y, sin_y = self._freqs(self.dy, y, self.theta)
        cos_x, sin_x = self._freqs(self.dx, x, self.theta)
        cos = torch.cat([cos_l, cos_y, cos_x], dim=-1)
        sin = torch.cat([sin_l, sin_y, sin_x], dim=-1)
        return cos, sin

    @staticmethod
    def apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q,k:  [B, heads, S, head_dim]
        cos/sin: [S, head_dim]
        """
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        # 经典旋转：把最后一维两两配对旋转
        def rotate_half(t):
            t1 = t[..., ::2]
            t2 = t[..., 1::2]
            return torch.stack([-t2, t1], dim=-1).flatten(-2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


# ---------------------------
# ViT blocks with RoPE attention
# ---------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, theta: float = 10000.0, rope_dims: Tuple[int,int,int] = None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = RoPE3D(self.head_dim, theta=theta, dims=rope_dims)

    def forward(self, x: torch.Tensor, ids_lyx: torch.Tensor):
        """
        x: [B, S, C], ids_lyx: [S, 3] = (layer, y, x)
        """
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, S, head_dim]

        cos, sin = self.rope.get_cos_sin(ids_lyx)     # [S, head_dim]
        q, k = RoPE3D.apply(q, k, cos, sin)

        # 使用 PyTorch SDPA（内部自己做 1/sqrt(d)）
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        x = attn_out.transpose(1, 2).reshape(B, S, C)
        x = self.proj_drop(self.proj(x))
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads=num_heads)   # rope_dims 自动适配
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, ids_lyx: torch.Tensor):
        x = x + self.attn(self.norm1(x), ids_lyx)
        x = x + self.mlp(self.norm2(x))
        return x



class ViTEncoder(nn.Module):
    def __init__(self, depth: int = 12, dim: int = 768, heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor, ids_lyx: torch.Tensor):
        for blk in self.blocks:
            x = blk(x, ids_lyx)
        return self.norm(x)

    def load_pretrained(self, arch: str = "vit-b/32", device: torch.device = None):
        """
        从 torchvision 加载 ImageNet 预训练 ViT 的 encoder 权重，映射到自定义 RoPE-ViT。
        仅拷贝: ln_1/ln_2, 自注意力(qkv/out_proj), MLP(fc1/fc2), 最后层norm。
        """
        if device is None:
            device = next(self.parameters()).device

        # 1) 拿到 torchvision 的 ViT
        from torchvision.models.vision_transformer import (
            vit_b_32, ViT_B_32_Weights,
            vit_h_14, ViT_H_14_Weights,
        )
        if arch == "vit-b/32":
            tv = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1).to(device).eval()
            expected_depth = 12
            expected_dim   = 768
            expected_heads = 12
        elif arch == "vit-h/14":
            tv = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1).to(device).eval()
            expected_depth = 32
            expected_dim   = 1280
            expected_heads = 16
        else:
            raise ValueError(f"Unknown arch: {arch}")

        assert len(self.blocks) == expected_depth and \
            self.blocks[0].attn.num_heads == expected_heads, \
            f"Your ViT config ({len(self.blocks)} layers, dim={self.blocks[0].norm1.normalized_shape[0]}) " \
            f"does not match torchvision {arch}"

        # 2) 逐层拷权重
        with torch.no_grad():
            for i, blk in enumerate(self.blocks):
                src = tv.encoder.layers[i]

                # norm1 / norm2
                blk.norm1.load_state_dict(src.ln_1.state_dict())
                blk.norm2.load_state_dict(src.ln_2.state_dict())

                # Attention: in_proj -> qkv, out_proj -> proj
                attn = src.self_attention               # nn.MultiheadAttention
                blk.attn.qkv.weight.copy_(attn.in_proj_weight)
                blk.attn.qkv.bias.copy_(attn.in_proj_bias)
                blk.attn.proj.weight.copy_(attn.out_proj.weight)
                blk.attn.proj.bias.copy_(attn.out_proj.bias)

                # MLP: torchvision 的 MLPBlock 是 [Linear, GELU, Dropout, Linear, Dropout]
                fc1 = src.mlp[0] if hasattr(src.mlp, "__getitem__") else src.mlp.fc1
                fc2 = src.mlp[3] if hasattr(src.mlp, "__getitem__") else src.mlp.fc2
                blk.mlp.fc1.weight.copy_(fc1.weight); blk.mlp.fc1.bias.copy_(fc1.bias)
                blk.mlp.fc2.weight.copy_(fc2.weight); blk.mlp.fc2.bias.copy_(fc2.bias)

            # final encoder norm
            self.norm.load_state_dict(tv.encoder.ln.state_dict())

        print(f"[ViT] Loaded pretrained encoder from torchvision {arch}.")
# ---------------------------
# Multi‑Layer Transparency Decoder
# ---------------------------

class MultiLayerTransparencyDecoder(nn.Module):
    """
    Input: concatenated latent tokens z of all segments (merged, background, fg_i)
           and a LayoutSpec describing per‑segment shapes/ids.

    Steps:
      1) Linearin: 16 -> hidden (e.g., 768)
      2) ViT with 3D‑RoPE on (x, y, layer)
      3) Linearout: hidden -> 256, reshape per segment into 8x8x4 patches
      4) Layout‑guided paste to return:
         - merged RGBA (alpha fixed 1.0),
         - background RGB (alpha 1.0),
         - list of foreground RGBA maps pasted into a full canvas or cropped region
    """
    def __init__(self, hidden_dim: int = 768, depth: int = 12, heads: int = 12, patch: int = 8):
        super().__init__()
        self.patch = patch
        self.linearin = nn.Linear(16, hidden_dim)
        self.vit = ViTEncoder(depth=depth, dim=hidden_dim, heads=heads, mlp_ratio=4.0)
        self.linearout = nn.Linear(hidden_dim, patch * patch * 4)

    def forward(self,
                z_concat: torch.Tensor,           # [B, sum_N, 16]
                layout: LayoutSpec,
                canvas_hw: Optional[Tuple[int,int]] = None
                ) -> Dict[str, torch.Tensor]:
        device = z_concat.device
        B, S, C = z_concat.shape
        assert C == 16

        # Build 3D indices for RoPE
        ids_lyx = layout.concat_positions_abs(device=device, patch=self.patch)  # [S, 3] (layer,y,x)
        assert ids_lyx.shape[0] == S, f"RoPE ids ({ids_lyx.shape[0]}) != tokens ({S}). Check z_concat order vs layout."

        x = self.linearin(z_concat)
        x = self.vit(x, ids_lyx)
        out = self.linearout(x)

        # Split back per segment and unpatchify
        seg_slices = layout.segment_token_slices(patch=self.patch)
        token_shapes = layout.token_shapes(patch=self.patch)
        outputs: Dict[str, torch.Tensor] = {}
        fg_full_rgba: List[torch.Tensor] = []

        # Determine canvas for paste
        if canvas_hw is None:
            # infer from the largest bbox or assume merged defines canvas
            # Prefer merged if present
            merged = next((s for s in layout.segments if s.kind == 'merged'), None)
            if merged is not None:
                canvas_hw = (merged.h, merged.w)
            else:
                # fall back to max bbox extent
                H = max(s.bbox_xyxy[3] if s.bbox_xyxy else s.h for s in layout.segments)
                W = max(s.bbox_xyxy[2] if s.bbox_xyxy else s.w for s in layout.segments)
                canvas_hw = (H, W)
        Hc, Wc = canvas_hw

        for seg in layout.segments:
            sl = seg_slices[seg.name]
            h8, w8 = token_shapes[seg.name]
            region = unpatchify_rgba(out[:, sl, :], h=h8 * self.patch, w=w8 * self.patch, patch=self.patch)
            # merged/background are RGB in the paper; keep RGBA for simplicity
            if seg.kind == 'foreground':
                # Paste into full‑size canvas at bbox
                assert seg.bbox_xyxy is not None, f"Foreground {seg.name} missing bbox"
                x0, y0, x1, y1 = seg.bbox_xyxy
                Ph, Pw = region.shape[-2:]
                assert (y1 - y0) == Ph and (x1 - x0) == Pw, \
                    f"Pred region {Ph}x{Pw} doesn't match bbox {(y1-y0)}x{(x1-x0)}"
                fg_canvas = torch.zeros((region.size(0), 4, Hc, Wc), device=region.device)
                fg_canvas[:, :, y0:y1, x0:x1] = region
                outputs[f"{seg.name}_rgba"] = region  # cropped
                fg_full_rgba.append(fg_canvas)
            elif seg.kind == 'merged':
                outputs['merged_rgba'] = region
            elif seg.kind == 'background':
                outputs['background_rgba'] = region

        if fg_full_rgba:
            outputs['foregrounds_full_rgba'] = torch.stack(fg_full_rgba, dim=1)  # [B, K, 4, H, W]
            # Compose a reference composite from bg + foregrounds
            if 'background_rgba' in outputs:
                bg = outputs['background_rgba']
                comp = bg.clone()
                comp_a = torch.ones_like(comp[:, 0:1])  # treat bg alpha as 1
                comp[:, 3:4] = comp_a
                for k in range(outputs['foregrounds_full_rgba'].size(1)):
                    fg = outputs['foregrounds_full_rgba'][:, k]
                    comp = alpha_composite(comp, fg)
                outputs['composed_from_layers'] = comp

        return outputs


def alpha_composite(dst_rgba: torch.Tensor, src_rgba: torch.Tensor) -> torch.Tensor:
    """Standard "over" alpha compositing. Inputs [B,4,H,W] in [-1,1] range -> returns same range.
    """
    # Convert to [0,1]
    def to01(x):
        return (x + 1.0) * 0.5
    def to11(x):
        return x * 2.0 - 1.0

    D = to01(dst_rgba)
    S = to01(src_rgba)
    Da, Sa = D[:, 3:4], S[:, 3:4]
    out_a = Sa + Da * (1 - Sa)
    out_rgb = (S[:, :3] * Sa + D[:, :3] * Da * (1 - Sa)) / (out_a.clamp(min=1e-6))
    out = torch.cat([out_rgb, out_a], dim=1)
    return to11(out)


# ---------------------------
# Minimal training wrapper (freeze EVAE, optimize decoder with L1)
# ---------------------------

class ARTDecoderModel(nn.Module):
    def __init__(self, decoder: Optional[MultiLayerTransparencyDecoder] = None):
        super().__init__()
        self.decoder = decoder or MultiLayerTransparencyDecoder()

    def forward(self, z_concat: torch.Tensor, layout: LayoutSpec, target: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        pred = self.decoder(z_concat, layout)
        # L1 losses on available targets
        loss = 0.0
        count = 0
        for key in [
            'merged_rgba', 'background_rgba'
        ]:
            if key in pred and key in target:
                loss = loss + F.l1_loss(pred[key], target[key])
                count += 1
        # per‑foreground cropped supervision
        for seg in layout.segments:
            if seg.kind == 'foreground':
                k = f"{seg.name}_rgba"
                if k in pred and k in target:
                    loss = loss + F.l1_loss(pred[k], target[k])
                    count += 1
        loss = loss / max(count, 1)
        return pred, loss


# ============================================================
# Example integration helpers
# ============================================================

def build_layout_from_masks(
    canvas_hw: Tuple[int, int],
    fg_entries: List[Tuple[str, Tuple[int,int,int,int]]],
) -> LayoutSpec:
    """
    canvas_hw: (H, W)
    fg_entries: list of (name, bbox_xyxy)
    layer ids: 0=merged, 1=background, 2.. for fg layers
    """
    H, W = canvas_hw
    # with bg
    segs = [
        SegmentSpec(name='merged', kind='merged', h=H, w=W, layer_id=0),
        SegmentSpec(name='background', kind='background', h=H, w=W, layer_id=1),
    ]
    lid = 2
    # without bg
    # segs = [
    #     SegmentSpec(name='merged', kind='merged', h=H, w=W, layer_id=0),
    # ]
    # lid = 1
    for name, (x0, y0, x1, y1) in fg_entries:
        h, w = (y1 - y0), (x1 - x0)
        segs.append(SegmentSpec(name=name, kind='foreground', h=h, w=w, layer_id=lid, bbox_xyxy=(x0,y0,x1,y1)))
        lid += 1
    return LayoutSpec(segs)


# ============================================================
# (Optional) Sanity check shapes
# ============================================================
if __name__ == "__main__":
    B = 2
    H, W = 512, 512
    patch = 8
    h8, w8 = H//patch, W//patch

    # fake latents per segment: merged, background, fg_0 crop 128x160
    z_merged = torch.randn(B, 16, h8, w8)
    z_bg = torch.randn(B, 16, h8, w8)
    z_fg0 = torch.randn(B, 16, (128//patch), (160//patch))

    z_concat = torch.cat([
        patchify_latent(z_merged),
        patchify_latent(z_bg),
        patchify_latent(z_fg0)
    ], dim=1)

    layout = build_layout_from_masks(
        canvas_hw=(H, W),
        fg_entries=[("fg_000", (100, 140, 260, 268))],
    )

    dec = MultiLayerTransparencyDecoder()
    out = dec(z_concat, layout, canvas_hw=(H, W))
    for k,v in out.items():
        print(k, tuple(v.shape))
