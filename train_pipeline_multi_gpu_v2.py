import os
import json
import math
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as T

# 本地模块
from utils import segment_by_boxes
from vae import Vae
from decoder import (
    MultiLayerTransparencyDecoder,
    build_layout_from_masks,
    patchify_latent,
)

# -----------------------------
# 分布式训练工具函数
# -----------------------------

def setup_distributed(rank, world_size, port="12355"):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # 设置CUDA设备
    torch.cuda.set_device(rank)
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 同步所有进程
    dist.barrier()

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    """获取当前进程rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """获取总进程数"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process():
    """判断是否为主进程"""
    return get_rank() == 0

def reduce_tensor(tensor):
    """跨所有进程对tensor进行平均"""
    if get_world_size() <= 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt

# -----------------------------
# 改进的工具函数
# -----------------------------

def extract_object_rgba_v2(
    image: Union[str, np.ndarray, Image.Image],
    mask: np.ndarray,
    save_path: str = None
) -> np.ndarray:
    """
    改进版RGBA提取：只复制mask区域的RGB，透明区域使用中性灰填充
    
    Args:
        image: 原始图像
        mask: 二值掩码 (0/1 或 False/True)
        save_path: 可选保存路径
        
    Returns:
        np.ndarray: RGBA图像 (H, W, 4)，透明区域为黑色(0,0,0)+透明
    """
    from PIL import Image as PILImage
    
    # 加载图像
    if isinstance(image, str):
        img_array = np.array(PILImage.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image.copy()
    
    # 确保掩码是布尔类型
    mask_bool = mask.astype(bool)
    
    # 创建RGBA图像（初始化为黑色+透明）
    h, w = img_array.shape[:2]
    rgba_image = np.full((h, w, 4), [0, 0, 0, 0], dtype=np.uint8)  # 黑色+透明

    # 只复制mask区域的RGB，透明区域保持黑色
    rgba_image[mask_bool, :3] = img_array[mask_bool]
    
    # 设置Alpha通道：掩码区域不透明(255)，其余透明(0)
    rgba_image[:, :, 3] = mask_bool.astype(np.uint8) * 255
    
    # 保存文件
    if save_path:
        img_pil = PILImage.fromarray(rgba_image, mode='RGBA')
        img_pil.save(save_path)
        if is_main_process():
            print(f"纯净透明物体图像保存到: {save_path} (透明区域: 黑色)")
    
    return rgba_image

def rgba_to_gray_rgb_paper_tensor11(rgba_u8: np.ndarray, device: str) -> torch.Tensor:
    """
    I_hat = (0.5*alpha + 0.5) * I_rgb, [-1,1] Tensor 
    Input: rgba_u8 [H,W,4] uint8
    Return: x_hat [1,3,H,W] float32 in [-1,1]
    """
    rgba = torch.from_numpy(rgba_u8.astype(np.float32) / 255.0).to(device)  # [H,W,4] in [0,1]
    rgb01 = rgba[..., :3]                            # [H,W,3]
    a01   = rgba[..., 3:4]                           # [H,W,1]
    rgb11 = rgb01 * 2.0 - 1.0                        # → [-1,1]
    a11 = a01 * 2.0 - 1.0
    scale = 0.5 * a11 + 0.5
    hat11 = scale * rgb11
    x_hat = hat11.permute(2,0,1).unsqueeze(0).contiguous()  # [1,3,H,W]
    return x_hat

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def assert_multiple_of(x: int, m: int, name: str):
    if x % m != 0:
        raise ValueError(f"{name}={x} 必须能被 {m} 整除（建议预处理或在本脚本里做 pad/crop）。")

def tight_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)

def align_bbox_to_multiple(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    multiple: int = 16,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    pad_w = new_w - w
    pad_h = new_h - h

    nx0 = max(0, x0 - pad_w // 2)
    ny0 = max(0, y0 - pad_h // 2)
    nx1 = min(img_w, nx0 + new_w)
    ny1 = min(img_h, ny0 + new_h)
    if (nx1 - nx0) < new_w: nx0 = max(0, nx1 - new_w)
    if (ny1 - ny0) < new_h: ny0 = max(0, ny1 - new_h)
    return int(nx0), int(ny0), int(nx1), int(ny1)

def rgba_to_gray_rgb(rgba: np.ndarray, gray=(127,127,127)) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    a = (rgba[..., 3:4].astype(np.float32) / 255.0)
    gray_arr = np.array(gray, dtype=np.float32) / 255.0
    out = a * rgb + (1.0 - a) * gray_arr
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out

def to_tensor_rgb_norm01(rgb_uint8: np.ndarray) -> torch.Tensor:
    # [H,W,3] uint8 -> [1,3,H,W] float in [0,1]
    x = torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    return x

def to_tensor_rgb_norm11(rgb_uint8: np.ndarray, device: str) -> torch.Tensor:
    tx = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    pil = Image.fromarray(rgb_uint8, mode="RGB")
    return tx(pil).unsqueeze(0).to(device)

def rgba_uint8_to_tensor_norm11(rgba_uint8: np.ndarray, device: str) -> torch.Tensor:
    # [H,W,4] uint8 -> [1,4,H,W] in [-1,1]
    x = torch.from_numpy(rgba_uint8.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
    x = x * 2.0 - 1.0
    return x

@torch.no_grad()
def encode_with_flux_vae(vae: Vae, rgb_uint8: np.ndarray) -> torch.FloatTensor:
    x = to_tensor_rgb_norm11(rgb_uint8, device=next(vae.parameters()).device)
    encoded = vae.flux_vae.encode(x)
    z = encoded.latent_dist.mode()  # [1,16,h8,w8]（Flux EVAE）
    return z

def crop_by_bbox(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x0,y0,x1,y1 = bbox
    return img[y0:y1, x0:x1, ...]

def save_image_rgba(t: torch.Tensor, path: str):
    # t: [1,4,H,W] or [4,H,W], in [-1,1]
    if t.dim() == 4: t = t[0]
    x = (t.clamp(-1,1).permute(1,2,0).cpu().numpy() + 1.0) * 127.5
    x = np.clip(x, 0, 255).astype(np.uint8)
    Image.fromarray(x, mode="RGBA").save(path)

# -----------------------------
# 数据集（batch_size=1）
# -----------------------------

def collate_fn(batch):
    """自定义collate函数，用于批处理数据"""
    return {
        "image": [item["image"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
    }

class ARTDataset(Dataset):
    """
    读取 jsonl，每行：
      {"image": "/abs/path/img.png",
       "boxes": {"name":[x1,y1,x2,y2], ...}}
    """
    def __init__(self, jsonl_path: str):
        super().__init__()
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                self.items.append(json.loads(line))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        image_path: str = rec["image"]
        boxes: Dict[str, List[int]] = rec.get("boxes", {})
        return {"image": image_path, "boxes": boxes}

# -----------------------------
# 改进版训练器（支持多GPU）
# -----------------------------

class ARTTrainer:
    def __init__(
        self,
        flux_vae_path: str,
        arch: str,
        device: str,
        lr: float,
        weight_decay: float,
        save_dir: str,
        lambda_fg: float = 1.0,
        lambda_merged: float = 0.5,
        lambda_composite: float = 1.0,
        lambda_bg: float = 0.0,
        use_amp: bool = True,
        distributed: bool = False,
    ):
        self.device = device
        self.distributed = distributed
        
        # 冻结编码器
        self.vae = Vae(model_path=flux_vae_path, device=device).eval().requires_grad_(False).to(device)
        
        # 解码器
        self.decoder = MultiLayerTransparencyDecoder(
            hidden_dim=768 if arch == "vit-b/32" else 1280,
            depth=12 if arch == "vit-b/32" else 32,
            heads=12 if arch == "vit-b/32" else 16,
            patch=8,
        ).to(device)

        self.decoder.vit.load_pretrained("vit-b/32", device=device)
        
        # 如果是分布式训练，包装模型
        if distributed:
            self.decoder = DDP(self.decoder, device_ids=[get_rank()], find_unused_parameters=False)
        
        # 优化器（只优化解码器）
        # 分布式训练时学习率需要根据world_size调整
        effective_lr = lr * get_world_size() if distributed else lr
        self.optimizer = optim.AdamW(self.decoder.parameters(), lr=effective_lr, weight_decay=weight_decay)
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=(use_amp and device.startswith("cuda")))
        self.use_amp = use_amp and device.startswith("cuda")

        # Loss 权重
        self.lambda_fg = lambda_fg
        self.lambda_merged = lambda_merged
        self.lambda_composite = lambda_composite
        self.lambda_bg = lambda_bg

        # 其他
        self.save_dir = save_dir
        if is_main_process():
            os.makedirs(self.save_dir, exist_ok=True)
        self.global_step = 0
        self.best_loss = float("inf")

    def _encode_full_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        bg = Image.open("/root/Huawei/data/room_bg.png").convert("RGB")
        W, H = img.size
        assert_multiple_of(H, 16, "H")
        assert_multiple_of(W, 16, "W")
        rgb = np.array(img)
        bg = np.array(bg)
        z_merged = encode_with_flux_vae(self.vae, rgb)
        z_bg = encode_with_flux_vae(self.vae, bg)
        # 目标 merged RGBA（alpha=1）
        rgba = np.concatenate([rgb, 255*np.ones((*rgb.shape[:2],1),dtype=np.uint8)], axis=2)
        bg_rgba = np.concatenate([bg, 255*np.ones((*bg.shape[:2],1),dtype=np.uint8)], axis=2)
        tgt_merged = rgba_uint8_to_tensor_norm11(rgba, self.device)  # [1,4,H,W]
        tgt_bg = rgba_uint8_to_tensor_norm11(bg_rgba, self.device)  # [1,4,H,W]
        return z_merged, z_bg, tgt_merged, tgt_bg, (H, W)

    def _build_foregrounds_v2(self, image_path: str, boxes: Dict[str, List[int]], canvas_hw: Tuple[int,int]):
        """
        改进版前景构建：使用纯净的RGBA提取，避免背景污染
        """
        H, W = canvas_hw
        # 1) SAM2 细化 mask（按 boxes）
        masks = segment_by_boxes(image=image_path, boxes_json=boxes, model_size="large", device=self.device)

        fg_entries: List[Tuple[str, Tuple[int,int,int,int]]] = []
        z_fg_list: List[torch.Tensor] = []
        tgt_fg_rgba: Dict[str, torch.Tensor] = {}

        for name, mask in masks.items():
            tb = tight_bbox_from_mask(mask)
            if tb is None:
                if is_main_process():
                    print(f"[WARN] {name}: 空掩码，跳过")
                continue
            bbox = align_bbox_to_multiple(tb, img_w=W, img_h=H, multiple=16)
            x0,y0,x1,y1 = bbox
            if (x1-x0)<=0 or (y1-y0)<=0:
                if is_main_process():
                    print(f"[WARN] {name}: 对齐后 bbox 非法，跳过")
                continue

            # 使用改进版RGBA提取（关键改进！）
            rgba_full = extract_object_rgba_v2(image_path, mask.astype(np.uint8))  # [H,W,4]
            rgba_crop = crop_by_bbox(rgba_full, bbox)                             # [h,w,4]
            
            # 现在rgba_crop中：
            # - mask区域：正常RGB + alpha=255
            # - 扩展区域：纯黑RGB(0,0,0) + alpha=0
            
            # 目标前景（裁剪）
            tgt_fg_rgba[name] = rgba_uint8_to_tensor_norm11(rgba_crop, self.device)  # [1,4,h,w]

            # 转论文版灰底（现在无背景污染）
            x_hat11 = rgba_to_gray_rgb_paper_tensor11(rgba_crop, self.device)  # [1,3,h,w], [-1,1]
            # 扩展区域：0.5 * (-1, -1, -1) = (-0.5, -0.5, -0.5) 深灰色，一致无污染

            # 直接编码
            with torch.no_grad():
                encoded = self.vae.flux_vae.encode(x_hat11)
                z_fg = encoded.latent_dist.mode()  # [1,16,h/8,w/8]

            z_fg_list.append(z_fg)
            fg_entries.append((name, bbox))

        return fg_entries, z_fg_list, tgt_fg_rgba

    def _build_tokens_and_layout(self, z_merged, z_bg, z_fg_list, fg_entries, canvas_hw):
        if z_bg is None:
            z_tokens = [patchify_latent(z_merged)]
        else:
            z_tokens = [patchify_latent(z_merged), patchify_latent(z_bg)]
        for z in z_fg_list: z_tokens.append(patchify_latent(z))
        z_concat = torch.cat(z_tokens, dim=1).to(self.device)  # [1,sum_N,16]
        layout = build_layout_from_masks(canvas_hw=canvas_hw, fg_entries=fg_entries)
        return z_concat, layout

    def _compute_loss(self, pred: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], fg_entries: List[Tuple[str,Tuple[int,int,int,int]]]):
        """
        pred 包含：
          - merged_rgba [1,4,H,W]
          - background_rgba [1,4,H,W]
          - {name}_rgba [1,4,h,w]（每个前景的裁剪）
          - foregrounds_full_rgba [1,K,4,H,W]
          - composed_from_layers [1,4,H,W]
        targets 包含：
          - merged_rgba [1,4,H,W]
          - {name}_rgba [1,4,h,w]
        """
        loss = 0.0
        # 前景裁剪 L1
        for name, _bbox in fg_entries:
            key = f"{name}_rgba"
            if key in pred and name in targets:
                loss_bg = nn.functional.l1_loss(pred[key], targets[name])
                loss = loss + self.lambda_fg * loss_bg
                
        # merged 与合成对 merged 的监督
        if "merged_rgba" in pred and "merged_rgba" in targets and self.lambda_merged > 0:
            loss = loss + self.lambda_merged * nn.functional.l1_loss(pred["merged_rgba"], targets["merged_rgba"])
        if "composed_from_layers" in pred and "merged_rgba" in targets and self.lambda_composite > 0:
            loss = loss + self.lambda_composite * nn.functional.l1_loss(pred["composed_from_layers"], targets["merged_rgba"])

        # 背景（默认不监督，若你有 GT 背景可开启）
        if "background_rgba" in pred and "background_rgba" in targets and self.lambda_bg > 0:
            loss = loss + self.lambda_bg * nn.functional.l1_loss(pred["background_rgba"], targets["background_rgba"])

        return loss

    def save_checkpoint(self, tag: str):
        # 只在主进程保存
        if not is_main_process():
            return
            
        path = os.path.join(self.save_dir, f"decoder_{tag}.pt")
        # 获取模型状态字典，处理DDP包装
        model_state = self.decoder.module.state_dict() if self.distributed else self.decoder.state_dict()
        torch.save({
            "step": self.global_step,
            "decoder": model_state,
            "best_loss": self.best_loss,
        }, path)
        print(f"[CKPT] saved to {path}")

    def train_one_epoch(self, loader: DataLoader, epoch: int, accum_steps: int = 1, log_interval: int = 20):
        self.decoder.train()
        running_loss_sum = 0.0     # 累积"逐样本的原始 loss"（未除以 B/accum）
        running_samples = 0        # 用于算"每样本平均损失"
        start = time.time()
        
        # 设置分布式采样器的epoch
        if self.distributed and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)

        for it, batch in enumerate(loader):
            image_paths: List[str] = batch["image"]
            boxes_list: List[Dict[str, List[int]]] = batch["boxes"]
            B = len(image_paths)
            assert B >= 1

            # 一个 DataLoader 批次内：对每个样本单独前向 + 反传（梯度会累积在同一模型上）
            for b in range(B):
                image_path = image_paths[b]
                boxes = boxes_list[b]

                # 1) 整图 merged/bg + merged target
                z_merged, z_bg, tgt_merged, tgt_bg, canvas_hw = self._encode_full_image(image_path)

                # 2) 前景（使用改进版函数）
                fg_entries, z_fg_list, tgt_fg_rgba = self._build_foregrounds_v2(image_path, boxes, canvas_hw)
                if len(fg_entries) == 0:
                    continue  # 当前样本跳过，但不影响本批次其他样本

                # 3) tokens + layout
                z_concat, layout = self._build_tokens_and_layout(z_merged, z_bg, z_fg_list, fg_entries, canvas_hw)
                
                # z_concat, layout = self._build_tokens_and_layout(z_merged, None, z_fg_list, fg_entries, canvas_hw)

                # 4) 前向 & loss（逐样本）
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pred = self.decoder(z_concat, layout, canvas_hw=canvas_hw)
                    targets = {"merged_rgba": tgt_merged, "background_rgba": tgt_bg}
                    targets.update(tgt_fg_rgba)
                    loss_b = self._compute_loss(pred, targets, fg_entries)  # 标准逐样本 loss
                    if is_main_process():
                        print(f"[Epoch {epoch}] | loss/sample {loss_b.item():.6f}")
                    # 为了实现 "有效 batch = B * accum_steps"
                    # 在反传时把每个样本的 loss 除以 (B * accum_steps)
                    loss_scaled = loss_b / (B * accum_steps)

                self.scaler.scale(loss_scaled).backward()

                # 统计/日志
                running_loss_sum += loss_b.item()
                running_samples += 1

            # —— 到这里，当前 DataLoader 批次里的 B 个样本都已经 backward 完成（梯度已累积）——
            # 每累计 accm_steps 个"DataLoader 批次"再 step 一次
            if (it + 1) % accum_steps == 0:
                # 在分布式训练中同步梯度
                if self.distributed:
                    # DDP会自动同步梯度，但我们需要等待所有进程完成
                    dist.barrier()
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            # 日志：打印"每样本平均损失"（只在主进程打印）
            if (it + 1) % log_interval == 0 and running_samples > 0 and is_main_process():
                elapsed = time.time() - start
                avg_per_sample = running_loss_sum / running_samples
                rank_info = f"[Rank {get_rank()}/{get_world_size()}] " if self.distributed else ""
                print(f"{rank_info}[Epoch {epoch}] iter {it+1}/{len(loader)} | step {self.global_step} | "
                    f"loss/sample {avg_per_sample:.4f} | {elapsed:.1f}s")
                running_loss_sum = 0.0
                running_samples = 0
                start = time.time()

    @torch.no_grad()
    def evaluate_one_batch(self, image_path: str, boxes: Dict[str, List[int]], tag: str = "val"):
        # 只在主进程进行评估
        if not is_main_process():
            return None
            
        self.decoder.eval()
        z_merged, z_bg, tgt_merged, tgt_bg, canvas_hw = self._encode_full_image(image_path)
        fg_entries, z_fg_list, tgt_fg_rgba = self._build_foregrounds_v2(image_path, boxes, canvas_hw)
        if len(fg_entries) == 0:
            print("[EVAL] no foreground, skip."); return None

        z_concat, layout = self._build_tokens_and_layout(z_merged, z_bg, z_fg_list, fg_entries, canvas_hw)
        # z_concat, layout = self._build_tokens_and_layout(z_merged, None, z_fg_list, fg_entries, canvas_hw)
        pred = self.decoder(z_concat, layout, canvas_hw=canvas_hw)

        out_dir = os.path.join(self.save_dir, f"vis_{tag}")
        os.makedirs(out_dir, exist_ok=True)
        if "merged_rgba" in pred: save_image_rgba(pred["merged_rgba"], os.path.join(out_dir, "merged_pred.png"))
        if "composed_from_layers" in pred: save_image_rgba(pred["composed_from_layers"], os.path.join(out_dir, "composed_pred.png"))
        
        for name, tgt_fg in tgt_fg_rgba.items():
            save_image_rgba(tgt_fg, os.path.join(out_dir, f"fg_{name}_target.png"))

        save_image_rgba(tgt_merged, os.path.join(out_dir, "merged_target.png"))
        if "background_rgba" in pred: save_image_rgba(pred["background_rgba"], os.path.join(out_dir, "background_pred.png"))

        if "foregrounds_full_rgba" in pred and pred["foregrounds_full_rgba"].size(1) > 0:
            for num in range(pred["foregrounds_full_rgba"].size(1)):
                save_image_rgba(pred["foregrounds_full_rgba"][:,num], os.path.join(out_dir, f"fg{num}_full_pred.png"))

        # # -------------------------
        # # 关键：bBox 内、mask 外（GT α=0）的预测 α 统计
        # # -------------------------
        # eps = 1e-6
        # overall_sum = 0.0
        # overall_cnt = 0
        # overall_leak_pix = 0
        # overall_leak_pix_total = 0

        # for name, tgt_fg in tgt_fg_rgba.items():
        #     key = f"{name}_rgba"
        #     if key not in pred:
        #         continue

        #     # [-1,1] → [0,1]
        #     tgt_a01  = (tgt_fg[:, 3:4]  + 1.0) * 0.5
        #     pred_a01 = (pred[key][:, 3:4] + 1.0) * 0.5

        #     # 在裁剪区域内，目标 α=0 的位置就是“mask 外”
        #     mask_out = (tgt_a01 <= eps)
        #     n = mask_out.sum().item()
        #     if n == 0:
        #         print(f"[EVAL/{tag}] {name}: no pixels in (bbox ∧ ~mask)")
        #         continue

        #     mean_alpha_out = pred_a01[mask_out].mean().item()
        #     # 也给个泄漏比例：预测 α > 0.05 的像素占比
        #     leak_ratio = (pred_a01[mask_out] > 0.05).float().mean().item()

        #     print(f"[EVAL/{tag}] {name}: mean α (bbox∧~mask) = {mean_alpha_out:.4f} | leak_ratio(>0.05) = {leak_ratio:.4f} | pix={n}")

        #     overall_sum += mean_alpha_out * n
        #     overall_cnt += n
        #     overall_leak_pix += (pred_a01[mask_out] > 0.05).float().sum().item()
        #     overall_leak_pix_total += n

        #     # （可选）对照：mask 内的 α 平均
        #     mask_in = (tgt_a01 >= 1.0 - eps)
        #     if mask_in.any():
        #         mean_alpha_in = pred_a01[mask_in].mean().item()
        #         print(f"[EVAL/{tag}] {name}: mean α (mask)     = {mean_alpha_in:.4f} | pix={mask_in.sum().item()}")

        # if overall_cnt > 0:
        #     overall_mean = overall_sum / overall_cnt
        #     overall_leak_ratio = overall_leak_pix / max(1, overall_leak_pix_total)
        #     print(f"[EVAL/{tag}] OVERALL: mean α (bbox∧~mask) = {overall_mean:.4f} | leak_ratio(>0.05) = {overall_leak_ratio:.4f} | pix={overall_cnt}")



        # 返回一个简单的 L1 评估
        val_loss = 0.0
        if "merged_rgba" in pred:
            val_loss += nn.functional.l1_loss(pred["merged_rgba"], tgt_merged).item()
        # if "composed_from_layers" in pred:
        #     val_loss += nn.functional.l1_loss(pred["composed_from_layers"], tgt_merged).item()
        return val_loss

# -----------------------------
# 主入口
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True, help="jsonl 列表（image, boxes）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)  # 建议保持 1
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--flux_vae_path", type=str, default="models/Flux_vae")
    ap.add_argument("--arch", type=str, default="vit-b/32", choices=["vit-b/32", "vit-h/14"])
    ap.add_argument("--save_dir", type=str, default="./checkpoints_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--eval_first_item", action="store_true", help="每个 epoch 末用第一条样本做一次可视化")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use (e.g., 'cpu', 'cuda')")
    # 分布式训练参数
    ap.add_argument("--distributed", action="store_true", help="启用分布式训练")
    ap.add_argument("--world_size", type=int, default=1, help="总GPU数量")
    ap.add_argument("--master_port", type=str, default="12355", help="主进程端口")
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader工作进程数")
    args = ap.parse_args()
    return args

def main_worker(rank, world_size, args):
    """单个GPU进程的主函数"""
    if args.distributed:
        setup_distributed(rank, world_size, args.master_port)
    
    # 设置种子，确保每个进程的随机性一致
    set_seed(args.seed + rank)

    device = f"cuda:{rank}" if args.distributed else args.device
    if is_main_process():
        print(f"[Device] {device}, Rank: {rank}/{world_size}")
        print(f"[Training] Distributed: {args.distributed}, World Size: {world_size}")
        print(f"[改进] 使用纯净RGBA提取，避免背景污染embedding")

    ds = ARTDataset(args.train_jsonl)
    
    # 分布式采样器
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank) if args.distributed else None
    
    # 分布式训练时建议将num_workers设置为0以避免pickle问题
    num_workers = 0 if args.distributed else args.num_workers
    
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None),  # 使用采样器时不能shuffle
        sampler=sampler,
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    trainer = ARTTrainer(
        flux_vae_path=args.flux_vae_path,
        arch=args.arch,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        lambda_fg=1.0,
        lambda_merged=1,
        lambda_composite=0,
        lambda_bg=1,
        use_amp=True,
        distributed=args.distributed,
    )

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        trainer.train_one_epoch(dl, epoch, accum_steps=args.accum_steps, log_interval=args.log_interval)

        # 简单评估：拿第一条样本可视化 + 报一个 L1（只在主进程）
        if args.eval_first_item and len(ds) > 0:
            first = ds[0]
            val_loss = trainer.evaluate_one_batch(first["image"], first["boxes"], tag=f"epoch{epoch}")
            if val_loss is not None:
                print(f"[Eval] epoch {epoch} val_loss ≈ {val_loss:.4f}")
                if val_loss < trainer.best_loss:
                    trainer.best_loss = val_loss
                    trainer.save_checkpoint("best")

        # 每个 epoch 末都存一份
        # trainer.save_checkpoint(f"epoch{epoch}")
        
        # 分布式同步
        if args.distributed:
            dist.barrier()

    if is_main_process():
        print("Training completed!")
    
    # 清理分布式环境
    if args.distributed:
        cleanup_distributed()

def main():
    args = parse_args()
    
    if args.distributed:
        # 多GPU分布式训练
        if args.world_size == 1:
            args.world_size = torch.cuda.device_count()
        
        if args.world_size <= 1:
            print("Warning: Distributed training requested but only 1 GPU available. Running single-GPU training.")
            args.distributed = False
            main_worker(0, 1, args)
        else:
            print(f"Starting distributed training on {args.world_size} GPUs")
            mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        # 单GPU训练
        main_worker(0, 1, args)

if __name__ == "__main__":
    main()