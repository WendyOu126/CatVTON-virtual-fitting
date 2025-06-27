# 上衣分割批量处理工具
# 功能：自动检测模特上衣区域，生成透明背景图片和掩码
# 支持格式：jpg, jpeg, png

import os
import cv2
import numpy as np
import warnings
from tqdm import tqdm
from PIL import Image
import paddlehub as hub
import time
import sys  # 新增系统模块

# 设置系统编码为UTF-8（解决中文路径问题）
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None


def top(input_images,output_results):
    warnings.filterwarnings("ignore", category=UserWarning)

    # 配置参数
    INPUT_DIR = input_images  # 输入文件夹路径（存放模特照片）
    OUTPUT_DIR = output_results  # 输出文件夹路径
    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")  # 支持的图片格式
    MODEL_NAME = "ace2p"  # 人体解析模型名称
    MAX_SIZE = 1024  # 最大图像尺寸（长边）
    UPPER_CLOTHES_ID = 5  # 上衣类别ID
    LOWWER_CLOTHES_ID = 9 # 下衣类别ID
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

    # 加载人体解析模型
    print(f"⏳ 正在加载人体解析模型 '{MODEL_NAME}'...")
    try:
        start_time = time.time()
        model = hub.Module(name=MODEL_NAME)
        load_time = time.time() - start_time
        print(f"✅ 模型加载成功 ({load_time:.2f}秒)")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("尝试使用备用模型 'humanseg'...")
        try:
            model = hub.Module(name="humanseg")
            print("✅ 备用模型加载成功")
        except Exception as e2:
            print(f"❌ 加载备用模型失败: {e2}")
            print("请检查网络连接或尝试重新安装: pip install --upgrade paddlepaddle paddlehub")
            exit(1)

    # 获取输入文件列表
    input_files = []
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(SUPPORTED_EXTS):
            input_files.append(file)

    if not input_files:
        print(f"❌ 在 '{INPUT_DIR}' 中未找到支持的图片文件")
        print(f"支持的格式: {', '.join(SUPPORTED_EXTS)}")
        exit(1)

    print(f"🔍 找到 {len(input_files)} 个待处理图片")
    # 处理所有图片
    success_count = 0
    failure_count = 0
    failure_messages = []

    print("🚀 开始批量处理图片...")
    start_time = time.time()

    # 使用tqdm创建进度条
    for file in tqdm(input_files, desc="处理进度", unit="图片"):
        input_path = os.path.join(INPUT_DIR, file)
        # 使用安全文件名处理（解决中文乱码）
        safe_prefix = file.split('.')[0]  # 直接使用原文件名（不含扩展名）

        success, message = process_image(
            input_path,
            safe_prefix,  # 使用修改后的参数
            MAX_SIZE,
            UPPER_CLOTHES_ID,
            model,
            OUTPUT_DIR
        )

        if success:
            success_count += 1
        else:
            failure_count += 1
            failure_messages.append(message)

    # 计算处理时间
    total_time = time.time() - start_time

    # 打印摘要报告
    print("\n" + "=" * 50)
    print(f"📊 处理摘要")
    print("=" * 50)
    print(f"✅ 成功处理: {success_count} 张图片")
    print(f"❌ 处理失败: {failure_count} 张图片")
    print(f"⏱️ 总耗时: {total_time:.2f}秒")
    print(f"📂 输出目录: {os.path.abspath(OUTPUT_DIR)}")

    if failure_count > 0:
        print("\n❌ 失败详情:")
        for msg in failure_messages:
            print(f"  - {msg}")

    # 显示文件夹结构
    print("\n📁 输出文件夹结构:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── masks/         # 上衣掩码（二值PNG）")

    print("\n✨ 处理完成！")

# 处理函数
def process_image(image_path, output_prefix, MAX_SIZE, UPPER_CLOTHES_ID, model, OUTPUT_DIR):
    try:
        # 1. 加载图像
        pil_img = Image.open(image_path)
        orig_width, orig_height = pil_img.size

        # 2. 调整大小（保持比例）
        scale_factor = 1.0
        if max(orig_width, orig_height) > MAX_SIZE:
            scale_factor = MAX_SIZE / max(orig_width, orig_height)
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

        # 3. 转换为OpenCV格式
        img = np.array(pil_img)
        if len(img.shape) == 2:  # 灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 4. 人体解析
        result = model.segmentation(images=[img])
        parsing = result[0]['data']

        # 5. 创建上衣掩码
        upper_mask = (parsing == UPPER_CLOTHES_ID).astype(np.uint8) * 255

        # 6. 提取透明背景的上衣
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = np.where(parsing == UPPER_CLOTHES_ID, 255, 0).astype(np.uint8)

        # 7. 创建预览图像
        preview_img = img.copy()
        preview_img[upper_mask == 255] = (0, 255, 0)  # 用绿色标记上衣区域

        # 8. 保存结果（使用PIL代替OpenCV解决文件名编码问题）
        mask_path = os.path.join(OUTPUT_DIR, "masks", f"{output_prefix}_mask.png")

        # 使用PIL保存掩码（解决中文乱码）
        mask_img = Image.fromarray(upper_mask)
        mask_img.save(mask_path)

        return True, f"处理成功: {os.path.basename(image_path)}"

    except Exception as e:
        return False, f"处理失败: {os.path.basename(image_path)} - {str(e)}"

input_images=r"C:\Users\ASUS\Desktop\PRDataSet\model"
output_images=r"C:\Users\ASUS\Desktop\PRDataSet\result"
top(input_images,output_images)