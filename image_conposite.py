import os
import json
from PIL import Image
import numpy as np

def load_json(json_path):
    """读取 JSON 文件，返回图像对的对应关系"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON 文件根元素必须是一个列表")
        return data
    except Exception as e:
        print(f"读取 JSON 文件 {json_path} 失败: {e}")
        return []

def save_json(json_path, data):
    """保存 JSON 文件"""
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"已更新 JSON 文件：{json_path}")
    except Exception as e:
        print(f"保存 JSON 文件 {json_path} 失败: {e}")

def composite_images(model_path, mask_path, output_path):
    """
    使用掩码保留模特非衣服部分，生成透明背景的图像
    参数：
        model_path: 模特图像路径
        mask_path: 掩码图像路径（0表示非衣服区域，255表示衣服区域）
        output_path: 输出合成图像路径（PNG 格式，支持透明度）
    """
    try:
        # 加载模特图像和掩码图像
        model_img = Image.open(model_path).convert('RGBA')  # 转换为 RGBA 以支持透明度
        mask_img = Image.open(mask_path).convert('L')  # 转换为灰度图

        # 确保掩码与模特图像尺寸一致
        if mask_img.size != model_img.size:
            mask_img = mask_img.resize(model_img.size, Image.NEAREST)

        # 将掩码转换为 numpy 数组
        mask_array = np.array(mask_img) / 255.0  # 归一化到 [0, 1]
        model_array = np.array(model_img)

        # 创建透明背景（RGBA 全零）
        composite_array = np.zeros_like(model_array, dtype=np.uint8)

        # 保留非衣服区域（mask=0），衣服区域（mask=1）设为透明
        # Alpha 通道：1 - mask_array（mask=0 时 Alpha=1，mask=1 时 Alpha=0）
        alpha_channel = (1 - mask_array) * 255
        composite_array[..., 0:3] = model_array[..., 0:3] * (1 - mask_array)[:, :, np.newaxis]  # RGB 只保留非衣服部分
        composite_array[..., 3] = alpha_channel  # Alpha 通道控制透明度

        # 保存为 PNG 以支持透明度
        composite_img = Image.fromarray(composite_array, mode='RGBA')
        composite_img.save(output_path, format='PNG')
        print(f"已保存非衣服部分图像到：{output_path}")
    except Exception as e:
        print(f"合成图像失败 (model: {model_path}, mask: {mask_path}): {e}")

def batch_composite(json_path, output_dir, base_dir="/root/autodl-tmp/CatVTON"):
    """批量处理模特图像和掩码图像的合成，并更新 JSON 文件"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取 JSON 文件
    image_groups = load_json(json_path)
    if not image_groups:
        print("没有有效的图像组，退出")
        return

    # 遍历图像组
    for group in image_groups:
        # 获取模特和掩码路径（忽略 clothes）
        model_paths = group.get('models', [])
        mask_paths = group.get('agnostic-mask', [])

        # 验证路径列表
        if not model_paths or not mask_paths:
            print(f"跳过：无效的路径组 (models: {model_paths}, masks: {mask_paths})")
            continue

        # 确保模特和掩码列表长度匹配
        if len(model_paths) != len(mask_paths):
            print(f"跳过：模特和掩码数量不匹配 (models: {len(model_paths)}, masks: {len(mask_paths)})")
            continue

        # 初始化 composites 列表（如果不存在）
        if 'composites' not in group:
            group['composites'] = []

        # 遍历模特和掩码对
        for model_path, mask_path in zip(model_paths, mask_paths):
            # 处理相对路径
            model_path = os.path.join(base_dir, model_path) if not os.path.isabs(model_path) else model_path
            mask_path = os.path.join(base_dir, mask_path) if not os.path.isabs(mask_path) else mask_path

            # 检查文件是否存在
            if not os.path.exists(model_path) or not os.path.exists(mask_path):
                print(f"跳过：模特图像 {model_path} 或掩码 {mask_path} 不存在")
                continue

            # 生成输出文件名
            model_name = os.path.basename(model_path).split('.')[0]
            output_path = os.path.join(output_dir, f"{model_name}_non_clothes.png")

            # 执行图像合成
            composite_images(model_path, mask_path, output_path)

            # 将合成图像的相对路径添加到 JSON
            relative_output_path = os.path.relpath(output_path, base_dir)
            group['composites'].append(relative_output_path)

        # 更新 JSON 文件
        save_json(json_path, image_groups)

if __name__ == "__main__":
    # 配置路径
    json_path = "/root/autodl-tmp/CatVTON/datasets/data_pairs.json"  # JSON 文件路径
    output_dir = "/root/autodl-tmp/CatVTON/datasets/output_composites"  # 输出目录
    base_dir = "/root/autodl-tmp/CatVTON/datasets"  # JSON 中路径的基目录

    # 运行批量合成
    batch_composite(json_path, output_dir, base_dir)