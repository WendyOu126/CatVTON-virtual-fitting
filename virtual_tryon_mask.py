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
from tqdm import tqdm

# 数据集类，加载模特图、衣服图和掩码
class HanscaTryonDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_root_path = args.data_root_path
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                               do_convert_grayscale=True)
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
                # 假设掩码文件命名为xxx_mask.png
                mask_path = composite_img.replace('_non_clothes.png', '_mask.png')
                assert os.path.exists(os.path.join(self.data_root_path, mask_path)), f"Mask file {mask_path} does not exist."
                data.append({
                    'person_name': person_name,
                    'cloth_name': cloth_name,
                    'person': os.path.join(self.data_root_path, model_img),
                    'cloth': os.path.join(self.data_root_path, cloth_img),
                    'mask': os.path.join(self.data_root_path, mask_path),
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        person = Image.open(data['person'])
        cloth = Image.open(data['cloth'])
        mask = Image.open(data['mask'])
        return {
            'index': idx,
            'person_name': data['person_name'],
            'cloth_name': data['cloth_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0],
        }

# 试衣函数
def tryon_mask(unet, vae, dataloader, noise_scheduler, accelerator, args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    pipeline = CatVTONPipeline(unet=unet, vae=vae, device=accelerator.device, noise_sheduler=noise_scheduler)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Virtual Try-On (Mask-Based)"):
            person_images = batch['person'].to(accelerator.device)
            cloth_images = batch['cloth'].to(accelerator.device)
            masks = batch['mask'].to(accelerator.device)
            
            results = pipeline(
                person_images,
                cloth_images,
                masks,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            )
            
            person_images_pil = numpy_to_pil(person_images)
            cloth_images_pil = numpy_to_pil(cloth_images)
            results_pil = numpy_to_pil(results)
            
            for j, result in enumerate(results_pil):
                person_name = batch['person_name'][j].split('.')[0]
                cloth_name = batch['cloth_name'][j].split('.')[0]
                output_path = os.path.join(output_dir, f"{person_name}_{cloth_name}_tryon.jpg")
                w, h = result.size
                concated_result = Image.new('RGB', (w * 3, h))
                concated_result.paste(person_images_pil[j], (0, 0))
                concated_result.paste(cloth_images_pil[j], (w, 0))
                concated_result.paste(result, (w * 2, 0))
                concated_result.save(output_path)
    
    del pipeline

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str,
                        default="/CatVTON/stable-diffusion-inpainting/")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained U-Net model")
    parser.add_argument('--data_root_path', type=str, default='/CatVTON/non_mask_train_data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--output_dir", type=str, default="tryon_output_mask")
    parser.add_argument("--vae_path", type=str, default="/CatVTON/sd-vae-ft-mse")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
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
    
    init_adapter(unet, SkipAttnProcessor)
    if args.model_path:
        unet.load_state_dict(torch.load(args.model_path), strict=False)
        print(f"Loaded model from {args.model_path}")

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        import xformers
        unet.enable_xformers_memory_efficient_attention()
    
    dataset = HanscaTryonDataset(args=args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    unet, dataloader = accelerator.prepare(unet, dataloader)
    
    tryon_mask(unet, vae, dataloader, noise_scheduler, accelerator, args)

if __name__ == '__main__':
    main()