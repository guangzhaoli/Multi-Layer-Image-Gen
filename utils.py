# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Union

# Add SAM2 path to system path
sam2_path = "/root/Huawei/segment-anything-2"
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


def segment_all_objects(
    image: Union[str, np.ndarray, Image.Image],
    model_size: str = "base_plus",
    device: str = "cuda",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 100,
    return_binary_masks: bool = True
) -> Union[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Use SAM2 to automatically segment all objects in an image
    
    Args:
        image: Input image, can be file path (str), numpy array or PIL Image
        model_size: Model size, options: "tiny", "small", "base_plus", "large"
        device: Device, "cuda" or "cpu"
        points_per_side: Number of sampling points per side, affects precision and speed
        pred_iou_thresh: IoU threshold to filter low-quality masks
        stability_score_thresh: Stability threshold to filter unstable masks
        min_mask_region_area: Minimum mask region area to filter small regions
        return_binary_masks: If True, return only binary masks; if False, return full info
    
    Returns:
        If return_binary_masks=True: List[np.ndarray] - List of binary masks (2D arrays)
        If return_binary_masks=False: List[Dict] - Full segmentation info including:
            - segmentation: segmentation mask
            - bbox: bounding box [x, y, width, height]
            - area: area
            - predicted_iou: predicted IoU
            - stability_score: stability score
    """
    
    # Load image
    if isinstance(image, str):
        image_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        image_array = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image_array = (image * 255).astype(np.uint8)
        else:
            image_array = image
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            pass  # Already HWC RGB format
        else:
            raise ValueError("Input numpy array must be in HWC RGB format")
    else:
        raise TypeError("Image must be file path (str), PIL Image, or numpy array")
    
    # Model config mapping
    model_configs = {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"), 
        "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt")
    }
    
    if model_size not in model_configs:
        raise ValueError(f"model_size must be one of {list(model_configs.keys())}")
    
    config_file, checkpoint_path = model_configs[model_size]
    
    # Build full paths
    config_path = os.path.join(sam2_path, config_file)
    ckpt_path = os.path.join(sam2_path, checkpoint_path)
    
    # Check if files exist
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_path}\n"
            f"Please run: cd {sam2_path}/checkpoints && ./download_ckpts.sh"
        )
    
    # Build SAM2 model
    sam2_model = build_sam2(config_file, ckpt_path, device=device)
    
    # Create automatic mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
    )
    
    # Generate masks for all objects
    masks_info = mask_generator.generate(image_array)
    
    # 额外过滤：确保掩码面积大于min_mask_region_area
    if min_mask_region_area > 0:
        filtered_masks_info = []
        for mask_info in masks_info:
            mask_area = np.sum(mask_info['segmentation'])
            if mask_area >= min_mask_region_area:
                filtered_masks_info.append(mask_info)
        masks_info = filtered_masks_info
        print(f"面积过滤: {len(masks_info)} 个掩码面积 >= {min_mask_region_area}")
    
    if return_binary_masks:
        # Extract only the binary masks
        binary_masks = [mask_info['segmentation'] for mask_info in masks_info]
        return binary_masks
    else:
        # Return full information
        return masks_info


def visualize_masks(
    image: Union[str, np.ndarray, Image.Image],
    masks: List[Dict[str, Any]],
    show_bbox: bool = True,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Visualize segmentation results
    
    Args:
        image: Original image
        masks: Mask list returned by segment_all_objects
        show_bbox: Whether to show bounding boxes
        alpha: Mask transparency
        
    Returns:
        np.ndarray: Visualization result image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import random
    
    # Load image
    if isinstance(image, str):
        img_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image.copy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Generate random colors
    colors = []
    for _ in range(len(masks)):
        color = [random.random(), random.random(), random.random(), alpha]
        colors.append(color)
    
    # Draw each mask
    for i, mask_info in enumerate(masks):
        mask = mask_info['segmentation']
        bbox = mask_info['bbox']
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask] = colors[i]
        ax.imshow(colored_mask)
        
        # Draw bounding box
        if show_bbox:
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add area label
            ax.text(
                bbox[0], bbox[1] - 5,
                f"Area: {mask_info['area']}",
                color='white', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7)
            )
    
    ax.set_title(f"SAM2 Segmentation Results - {len(masks)} objects detected")
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    try:
        result = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
    except AttributeError:
        # Fallback for older matplotlib versions
        result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return result


def visualize_binary_masks(
    image: Union[str, np.ndarray, Image.Image],
    binary_masks: List[np.ndarray],
    alpha: float = 0.5,
    save_path: str = None
) -> np.ndarray:
    """
    Directly visualize binary masks with random colors
    
    Args:
        image: Original image
        binary_masks: List of binary masks (2D boolean arrays)
        alpha: Mask transparency (0-1)
        save_path: Optional path to save the result
        
    Returns:
        np.ndarray: Visualization result image (RGB)
    """
    import matplotlib.pyplot as plt
    import random
    
    # Load image
    if isinstance(image, str):
        img_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image.copy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Generate random colors for each mask
    for mask in binary_masks:
        # Generate random color
        color = [random.random(), random.random(), random.random(), alpha]
        
        # Create colored mask overlay
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask] = color
        ax.imshow(colored_mask)
    
    ax.set_title(f"Binary Masks Visualization - {len(binary_masks)} objects")
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    try:
        result = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
    except AttributeError:
        result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()
    return result


def create_mask_overlay(
    image: Union[str, np.ndarray, Image.Image],
    binary_masks: List[np.ndarray],
    colors: List[tuple] = None,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create mask overlay directly on image without matplotlib
    
    Args:
        image: Original image
        binary_masks: List of binary masks
        colors: Optional list of RGB colors for each mask
        alpha: Overlay transparency
        
    Returns:
        np.ndarray: Image with mask overlay (RGB, uint8)
    """
    # Load image
    if isinstance(image, str):
        img_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image.copy()
    
    result = img_array.astype(np.float32)
    
    # Generate colors if not provided
    if colors is None:
        np.random.seed(42)  # For reproducible colors
        colors = []
        for _ in range(len(binary_masks)):
            colors.append((
                np.random.randint(0, 255),
                np.random.randint(0, 255), 
                np.random.randint(0, 255)
            ))
    
    # Apply each mask
    for mask, color in zip(binary_masks, colors):
        mask_indices = mask > 0
        for c in range(3):
            result[mask_indices, c] = (1 - alpha) * result[mask_indices, c] + alpha * color[c]
    
    return result.astype(np.uint8)


def visualize_individual_masks(
    binary_masks: List[np.ndarray],
    save_dir: str = None,
    prefix: str = "mask"
) -> List[np.ndarray]:
    """
    对每个二值掩码单独可视化，显示为黑白图像
    
    Args:
        binary_masks: 二值掩码列表
        save_dir: 保存目录，如果提供则保存每个掩码
        prefix: 文件名前缀
        
    Returns:
        List[np.ndarray]: 每个掩码的黑白可视化图像列表 (0=黑色, 1=白色)
    """
    import matplotlib.pyplot as plt
    import os
    
    visualized_masks = []
    
    for i, mask in enumerate(binary_masks):
        # 将布尔掩码转换为0-255的灰度图像
        # True (1) -> 255 (白色), False (0) -> 0 (黑色)
        mask_image = (mask.astype(np.uint8) * 255)
        visualized_masks.append(mask_image)
        
        # 如果指定了保存目录，则保存每个掩码
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # 使用matplotlib保存，确保黑白显示
            plt.figure(figsize=(8, 8))
            plt.imshow(mask_image, cmap='gray', vmin=0, vmax=255)
            plt.title(f'Mask {i+1} - Area: {np.sum(mask)} pixels')
            plt.axis('off')
            
            save_path = os.path.join(save_dir, f"{prefix}_{i+1:03d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Saved mask {i+1} to {save_path}")
    
    return visualized_masks


def save_masks_as_images(
    binary_masks: List[np.ndarray], 
    save_dir: str,
    prefix: str = "mask"
) -> None:
    """
    直接将二值掩码保存为图像文件（更快的方法）
    
    Args:
        binary_masks: 二值掩码列表
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    import os
    from PIL import Image
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i, mask in enumerate(binary_masks):
        # 转换为0-255的灰度图像
        mask_image = (mask.astype(np.uint8) * 255)
        
        # 直接使用PIL保存
        img = Image.fromarray(mask_image, mode='L')  # 'L' 表示灰度模式
        save_path = os.path.join(save_dir, f"{prefix}_{i+1:03d}.png")
        img.save(save_path)
        
        print(f"Saved mask {i+1} to {save_path} (area: {np.sum(mask)} pixels)")


def segment_by_boxes(
    image: Union[str, np.ndarray, Image.Image],
    boxes_json: Union[str, Dict],
    model_size: str = "large",
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    根据VLA提供的边界框坐标使用SAM2分割物体
    
    Args:
        image: 输入图像
        boxes_json: 边界框JSON，格式如：
                   {"box1": [x1, y1, x2, y2], "box2": [x1, y1, x2, y2]}
                   或JSON字符串
        model_size: SAM2模型大小
        device: 设备选择
        
    Returns:
        Dict[str, np.ndarray]: 每个box对应的分割掩码
                              {"box1": mask1, "box2": mask2, ...}
    """
    import json
    
    # 处理JSON输入
    if isinstance(boxes_json, str):
        boxes = json.loads(boxes_json)
    else:
        boxes = boxes_json
    
    # 加载图像
    if isinstance(image, str):
        image_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        image_array = np.array(image.convert("RGB"))
    else:
        image_array = image
    
    # 模型配置
    model_configs = {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"), 
        "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt")
    }
    
    config_file, checkpoint_path = model_configs[model_size]
    ckpt_path = os.path.join(sam2_path, checkpoint_path)
    
    # 构建SAM2模型
    sam2_model = build_sam2(config_file, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # 设置图像
    predictor.set_image(image_array)
    
    results = {}
    
    # print(f"开始分割 {len(boxes)} 个边界框...")
    
    for box_name, box_coords in boxes.items():
        x1, y1, x2, y2 = box_coords
        
        # SAM2需要的边界框格式: [x1, y1, x2, y2]
        input_box = np.array([x1, y1, x2, y2])
       
        # 使用边界框作为提示进行分割
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # 添加batch维度
            multimask_output=False,   # 只输出最佳掩码
        )
        
        # 获取最佳掩码
        best_mask = masks[0]  # shape: (H, W)
        
        results[box_name] = best_mask.astype(np.uint8)
        
        mask_area = np.sum(best_mask)
        # print(f"  {box_name}: 坐标{box_coords} → 掩码面积{mask_area}像素")
    
    return results


def visualize_box_results(
    image: Union[str, np.ndarray, Image.Image],
    box_masks: Dict[str, np.ndarray],
    boxes_json: Union[str, Dict],
    save_path: str = None
) -> np.ndarray:
    """
    可视化基于边界框的分割结果
    
    Args:
        image: 原始图像
        box_masks: segment_by_boxes返回的掩码字典
        boxes_json: 原始边界框JSON
        save_path: 保存路径
        
    Returns:
        np.ndarray: 可视化结果图像
    """
    import json
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import random
    
    # 处理输入
    if isinstance(boxes_json, str):
        boxes = json.loads(boxes_json)
    else:
        boxes = boxes_json
        
    if isinstance(image, str):
        img_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image.copy()
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # 为每个物体分配颜色
    colors = {}
    for box_name in box_masks.keys():
        colors[box_name] = [random.random(), random.random(), random.random()]
    
    # 绘制掩码和边界框
    for box_name, mask in box_masks.items():
        # 绘制掩码
        colored_mask = np.zeros((*mask.shape, 4))
        color = colors[box_name] + [0.6]  # 添加透明度
        colored_mask[mask > 0] = color
        ax.imshow(colored_mask)
        
        # 绘制边界框
        x1, y1, x2, y2 = boxes[box_name]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, 
            edgecolor=colors[box_name], 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(
            x1, y1-5, 
            f"{box_name}",
            color='white', 
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[box_name], alpha=0.8)
        )
    
    ax.set_title(f"SAM2 边界框分割结果 - {len(box_masks)} 个物体")
    ax.axis('off')
    plt.tight_layout()
    
    # 转换为numpy数组
    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果保存到: {save_path}")
    
    plt.close()
    return result


def extract_object_rgba(
    image: Union[str, np.ndarray, Image.Image],
    mask: np.ndarray,
    save_path: str = None
) -> np.ndarray:
    """
    根据掩码提取物体，生成透明背景的RGBA图像
    
    Args:
        image: 原始图像
        mask: 二值掩码 (0/1 或 False/True)
        save_path: 可选保存路径
        
    Returns:
        np.ndarray: RGBA图像 (H, W, 4)，掩码外区域完全透明
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
    
    # 创建RGBA图像
    h, w = img_array.shape[:2]
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    # 复制RGB通道
    rgba_image[:, :, :3] = img_array
    
    # 设置Alpha通道：掩码区域不透明(255)，其余透明(0)
    rgba_image[:, :, 3] = mask_bool.astype(np.uint8) * 255
    
    # 保存文件
    if save_path:
        img_pil = PILImage.fromarray(rgba_image, mode='RGBA')
        img_pil.save(save_path)
        print(f"透明物体图像保存到: {save_path}")
    
    return rgba_image


def extract_multiple_objects_rgba(
    image: Union[str, np.ndarray, Image.Image],
    masks_dict: Dict[str, np.ndarray],
    save_dir: str = None
) -> Dict[str, np.ndarray]:
    """
    批量提取多个物体的透明RGBA图像
    
    Args:
        image: 原始图像
        masks_dict: 掩码字典 {"name": mask_array}
        save_dir: 保存目录
        
    Returns:
        Dict[str, np.ndarray]: 每个物体的RGBA图像字典
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    for obj_name, mask in masks_dict.items():
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"{obj_name}_rgba.png")
        
        rgba_img = extract_object_rgba(image, mask, save_path)
        results[obj_name] = rgba_img
        
        # 统计信息
        obj_pixels = np.sum(mask > 0)
        print(f"{obj_name}: {obj_pixels} 像素")
    
    return results


def create_composite_image(
    rgba_objects: List[np.ndarray],
    background_color: tuple = (255, 255, 255, 255),
    save_path: str = None
) -> np.ndarray:
    """
    将多个RGBA物体合成到新背景上
    
    Args:
        rgba_objects: RGBA图像列表
        background_color: 背景颜色 (R, G, B, A)
        save_path: 保存路径
        
    Returns:
        np.ndarray: 合成后的RGBA图像
    """
    if not rgba_objects:
        raise ValueError("需要至少一个RGBA物体")
    
    # 获取图像尺寸
    h, w = rgba_objects[0].shape[:2]
    
    # 创建背景
    composite = np.full((h, w, 4), background_color, dtype=np.uint8)
    
    # 逐层合成物体
    for rgba_obj in rgba_objects:
        # Alpha混合
        alpha = rgba_obj[:, :, 3:4] / 255.0
        composite[:, :, :3] = (1 - alpha) * composite[:, :, :3] + alpha * rgba_obj[:, :, :3]
        
        # 更新Alpha通道（取最大值）
        composite[:, :, 3] = np.maximum(composite[:, :, 3], rgba_obj[:, :, 3])
    
    # 保存结果
    if save_path:
        from PIL import Image as PILImage
        PILImage.fromarray(composite, mode='RGBA').save(save_path)
        print(f"合成图像保存到: {save_path}")
    
    return composite


if __name__ == "__main__":
    test_image = "/home/lgz/Code/ART/room.png"
    
    # 选择测试模式
    mode = "boxes"  # "auto" 或 "boxes"
    
    if mode == "auto":
        # 自动分割模式
        print("运行SAM2自动分割...")
        binary_masks = segment_all_objects(
            image=test_image,
            model_size="large",           
            device="cpu",                 
            points_per_side=32,          
            pred_iou_thresh=0.8,         
            stability_score_thresh=0.95, 
            min_mask_region_area=5000,    
            return_binary_masks=True     
        )
        
        print(f"检测到 {len(binary_masks)} 个物体")
        
        # 保存单独掩码
        save_masks_as_images(
            binary_masks,
            save_dir="/home/lgz/Code/ART/output_masks",
            prefix="mask"
        )
        
        # 创建可视化
        result_image = create_mask_overlay(
            test_image,
            binary_masks,
            alpha=0.6
        )
        
        # 保存结果
        from PIL import Image as PILImage
        PILImage.fromarray(result_image).save("/home/lgz/Code/ART/result.jpg")
        
        print(f"结果保存:")
        print(f"• 单独掩码: output_masks/ ({len(binary_masks)} 个文件)")
        print("• 综合可视化: result.jpg")
        
    elif mode == "boxes":
        # VLA边界框分割模式
        print("运行SAM2边界框分割...")
        
        # 示例VLA输出的边界框JSON
        vla_boxes = {
            "floor_lamp": [19, 145, 120, 406],
            "potted_plant": [627, 173, 723, 400]
        }
        
        # 基于边界框分割
        box_masks = segment_by_boxes(
            image=test_image,
            boxes_json=vla_boxes,
            model_size="large",
            device="cpu"
        )
        
        # 保存每个物体的单独掩码
        from PIL import Image as PILImage
        for box_name, mask in box_masks.items():
            save_path = f"/home/lgz/Code/ART/mask/{box_name}_mask.png"
            mask_image = (mask * 255).astype(np.uint8)
            PILImage.fromarray(mask_image, mode='L').save(save_path)
            print(f"保存 {box_name} 掩码到: {save_path}")
        
        # 新功能：提取透明背景物体
        print("\n提取透明背景物体...")
        rgba_objects = extract_multiple_objects_rgba(
            image=test_image,
            masks_dict=box_masks,
            save_dir="/home/lgz/Code/ART/rgba_objects"
        )
        
        # 创建合成图像（可选）
        print("\n创建合成图像...")
        rgba_list = list(rgba_objects.values())
        composite = create_composite_image(
            rgba_objects=rgba_list,
            background_color=(240, 240, 240, 255),  # 浅灰背景
            save_path="/home/lgz/Code/ART/composite.png"
        )
        
        # 创建可视化
        result_vis = visualize_box_results(
            image=test_image,
            box_masks=box_masks,
            boxes_json=vla_boxes,
            save_path="/home/lgz/Code/ART/box_result.jpg"
        )
        
        print(f"\n边界框分割结果:")
        print(f"• 单独掩码: mask/ ({len(box_masks)} 个文件)")
        print(f"• 透明物体: rgba_objects/ ({len(rgba_objects)} 个RGBA文件)")
        print("• 合成图像: composite.png")
        print("• 可视化结果: box_result.jpg")
        print(f"• 输入边界框: {vla_boxes}")