import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 配置4张RTX 3090 GPU
import torch
import argparse
import json
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import numpy as np
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from model.myPipeline import CatVTONPipeline
from diffusers.utils.import_utils import is_xformers_available
from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import numpy_to_pil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch_fidelity
import pandas as pd
from tqdm import tqdm

# 数据集类，适配.json格式的三元组数据
class HanscaEvalDataset(Dataset):
    def __init__(self, args, is_mask_based=True):
        self.args = args
        self.data_root_path = args.data_root_path
        self.is_mask_based = is_mask_based  # 是否基于掩码模型
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                               do_convert_grayscale=True) if is_mask_based else None
        self.data = self.load_data()

    def load_data(self):
        pair_txt = os.path.join(self.data_root_path, 'test_unpairs.json')
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            json_data = json.load(f)
        
        data = []
        self.data_root_path = os.path.join(self.data_root_path, "test")
        
        for item in json_data:
            cloth_img = item['clothes'][0]
            for model_img, composite_img in zip(item['models'], item['composites']):
                person_name = os.path.basename(model_img)
                cloth_name = os.path.basename(cloth_img)
                composite_name = os.path.basename(composite_img)
                data.append({
                    'person_name': person_name,
                    'cloth_name': cloth_name,
                    'composite_name': composite_name,
                    'person': os.path.join(self.data_root_path, model_img),
                    'cloth': os.path.join(self.data_root_path, cloth_img),
                    'composite': os.path.join(self.data_root_path, composite_img),
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        person = Image.open(data['person'])
        cloth = Image.open(data['cloth'])
        composite = Image.open(data['composite'])
        item = {
            'index': idx,
            'person_name': data['person_name'],
            'cloth_name': data['cloth_name'],
            'composite_name': data['composite_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'composite': self.vae_processor.preprocess(composite, self.args.height, self.args.width)[0],
        }
        if self.is_mask_based:
            # 假设composites文件名为xxx_non_clothes.png，需替换为xxx_mask.png
            mask_path = data['composite'].replace('_non_clothes.png', '_mask.png')
            assert os.path.exists(mask_path), f"Mask file {mask_path} does not exist."
            mask = Image.open(mask_path)
            item['mask'] = self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        return item

# 计算SSIM和PSNR
def compute_metrics(generated, target):
    generated = generated.cpu().permute(0, 2, 3, 1).float().numpy()
    target = target.cpu().permute(0, 2, 3, 1).float().numpy()
    generated = (generated * 255).round().astype(np.uint8)
    target = (target * 255).round().astype(np.uint8)
    
    ssim_values = []
    psnr_values = []
    for i in range(generated.shape[0]):
        gen_img = generated[i]
        tgt_img = target[i]
        if gen_img.shape[-1] == 1:
            gen_img = gen_img.squeeze()
            tgt_img = tgt_img.squeeze()
        ssim_val = ssim(gen_img, tgt_img, multichannel=True, data_range=255) if gen_img.ndim == 3 else ssim(gen_img, tgt_img, data_range=255)
        psnr_val = psnr(gen_img, tgt_img, data_range=255)
        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)
    return ssim_values, psnr_values

# 评估函数
def evaluate(unet, vae, dataloader, noise_scheduler, accelerator, args, is_mask_based=True):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    pipeline = CatVTONPipeline(unet=unet, vae=vae, device=accelerator.device, noise_sheduler=noise_scheduler)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    ssim_list = []
    psnr_list = []
    generated_images = []
    target_images = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            person_images = batch['person'].to(accelerator.device)
            cloth_images = batch['cloth'].to(accelerator.device)
            target_images_batch = batch['composite'].to(accelerator.device)
            
            # 生成试衣图像
            results = pipeline(
                person_images,
                cloth_images,
                batch['mask'].to(accelerator.device) if is_mask_based else None,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            )
            
            # 计算SSIM和PSNR
            ssim_vals, psnr_vals = compute_metrics(results, target_images_batch)
            ssim_list.extend(ssim_vals)
            psnr_list.extend(psnr_vals)
            
            # 保存生成图像用于FID计算
            generated_images.extend(numpy_to_pil(results))
            target_images.extend(numpy_to_pil(target_images_batch))
            
            # 保存可视化结果
            if args.concat_eval_results:
                person_images_pil = numpy_to_pil(person_images)
                cloth_images_pil = numpy_to_pil(cloth_images)
                for j, result in enumerate(numpy_to_pil(results)):
                    person_name = batch['person_name'][j].split('.')[0]
                    cloth_name = batch['cloth_name'][j].split('.')[0]
                    output_path = os.path.join(output_dir, f"{person_name}_{cloth_name}_eval.jpg")
                    w, h = result.size
                    concated_result = Image.new('RGB', (w * 4, h))
                    concated_result.paste(person_images_pil[j], (0, 0))
                    concated_result.paste(cloth_images_pil[j], (w, 0))
                    concated_result.paste(result, (w * 2, 0))
                    concated_result.paste(numpy_to_pil(target_images_batch)[j], (w * 3, 0))
                    concated_result.save(output_path)
    
    # 计算FID
    fid_score = torch_fidelity.calculate_metrics(
        input1=generated_images,
        input2=target_images,
        cuda=True,
        fid=True,
        input1_model='inception-v3'
    )['frechet_inception_distance']
    
    # 保存评估结果
    results = {
        'ssim': ssim_list,
        'psnr': psnr_list,
        'ssim_mean': np.mean(ssim_list),
        'psnr_mean': np.mean(psnr_list),
        'fid': fid_score
    }
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 打印平均指标
    print(f"Average SSIM: {results['ssim_mean']:.4f}")
    print(f"Average PSNR: {results['psnr_mean']:.4f}")
    print(f"FID: {results['fid']:.4f}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str,
                        default="/CatVTON/stable-diffusion-inpainting/")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained U-Net model")
    parser.add_argument('--data_root_path', type=str, default='/CatVTON/datasets')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--vae_path", type=str, default="/CatVTON/sd-vae-ft-mse")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--is_mask_based", action="store_true", help="Evaluate mask-based model")
    parser.add_argument("--concat_eval_results", default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(accelerator.device, dtype=weight_dtype)
    noise_scheduler = DDIMScheduler.from_pretrained(args.base_model_path + "scheduler")
    unet = UNet2DConditionModel.from_pretrained(args.base_model_path + "unet", use_safetensors=False).to(accelerator.device,
                                                                                  dtype=weight_dtype)
    
    # 调整U-Net输入通道（无需掩码模型）
    if not args.is_mask_based:
        unet.conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, kernel_size=3, padding=1).to(accelerator.device, dtype=weight_dtype)
    
    init_adapter(unet, SkipAttnProcessor)
    if args.model_path:
        unet.load_state_dict(torch.load(args.model_path), strict=False)
        print(f"Loaded model from {args.model_path}")

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        import xformers
        unet.enable_xformers_memory_efficient_attention()
    
    dataset = HanscaEvalDataset(args=args, is_mask_based=args.is_mask_based)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    unet, dataloader = accelerator.prepare(unet, dataloader)
    
    evaluate(unet, vae, dataloader, noise_scheduler, accelerator, args, is_mask_based=args.is_mask_based)

if __name__ == '__main__':
    main()