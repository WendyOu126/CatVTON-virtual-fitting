import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from PIL import Image, ImageFilter
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from model.myPipeline import CatVTONPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)
import random
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs as DDPK
import json

def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    if mask_np.ndim == 2:
        mask_np = mask_np[:, :, None]
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

class HanscaDataset(Dataset):
    def __init__(self, args, is_test=False, is_paired=True):
        self.args = args
        self.data_root_path = args.data_root_path
        self.is_test = is_test
        self.is_paired = is_paired
        assert is_test or is_paired
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                                do_convert_grayscale=True)
        self.data = self.load_data()

    def load_data(self):
        # 根据 is_test 加载对应的 .json 文件
        pair_txt = os.path.join(self.data_root_path, 'test_unpairs.json' if self.is_test else 'train_pairs.json')
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            json_data = json.load(f)
        
        data = []
        self.data_root_path = os.path.join(self.data_root_path, "test" if self.is_test else "train")
        
        # 解析 .json 文件中的 clothes、models 和 composites
        for item in json_data:
            cloth_img = item['clothes'][0]  # 取第一个服装图像
            for model_img, mask_img in zip(item['models'], item['composites']):
                # 提取文件名用于输出
                person_name = os.path.basename(model_img)
                cloth_name = os.path.basename(cloth_img)
                mask_name = os.path.basename(mask_img)
                data.append({
                    'person_name': person_name,
                    'cloth_name': cloth_name,
                    'mask_name': mask_name,
                    'person': os.path.join(self.data_root_path, model_img),
                    'cloth': os.path.join(self.data_root_path, cloth_img),
                    'mask': os.path.join(self.data_root_path, mask_img),
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        person, cloth, mask = [Image.open(data[key]) for key in ['person', 'cloth', 'mask']]
        return {
            'index': idx,
            'person_name': data['person_name'],
            'cloth_name': data['cloth_name'],
            'mask_name': data['mask_name'], 
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str,
                        default="/CatVTON/stable-diffusion-inpainting/")
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--data_root_path', type=str, default='/CatVTON/datasets/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--save_path", type=str, default="hansca_ckpts")
    parser.add_argument("--output_dir", type=str, default="hansca_output")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--start_epochs", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_epoch_freq", type=int, default=5)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])  # 默认启用 fp16
    parser.add_argument("--vae_path", type=str, default="/CatVTON/sd-vae-ft-mse")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", default=True)
    parser.add_argument("--use_tf32", default=True)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--concat_eval_results", default=True)
    parser.add_argument("--repaint", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

def val(unet, vae, val_dataloader, noise_scheduler, epoch, accelerator, args):
    output_dir = os.path.join(args.output_dir, "epoch_" + str(epoch))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        pipeline = CatVTONPipeline(unet=unet, vae=vae, device=accelerator.device, noise_sheduler=noise_scheduler)
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        for i, batch in enumerate(val_dataloader):
            if i >= 1:
                break
            person_images = batch['person']
            cloth_images = batch['cloth']
            masks = batch['mask']
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
            if args.concat_eval_results or args.repaint:
                person_images = to_pil_image(person_images)
                cloth_images = to_pil_image(cloth_images)
                masks = to_pil_image(masks)
            for j, result in enumerate(results):
                person_name = batch['person_name'][j].split('.')[0]
                cloth_name = batch['cloth_name'][j].split('.')[0]
                output_path = os.path.join(output_dir, f"{person_name}_{cloth_name}.jpg")
                if args.repaint:
                    person_path = os.path.join(args.data_root_path, batch['person_name'][j])
                    mask_path = os.path.join(args.data_root_path,-batch['mask_name'][j])
                    person_image = Image.open(person_path).resize(result.size, Image.LANCZOS)
                    mask = Image.open(mask_path).resize(result.size, Image.NEAREST)
                    result = repaint(person_image, mask, result)
                if args.concat_eval_results:
                    w, h = result.size
                    concated_result = Image.new('RGB', (w * 3, h))
                    concated_result.paste(person_images[j], (0, 0))
                    concated_result.paste(cloth_images[j], (w, 0))
                    concated_result.paste(result, (w * 2, 0))
                    result = concated_result
                result.save(output_path)
    del pipeline

def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    kwargs = DDPK(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.save_path is not None:
            os.makedirs(args.save_path, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.use_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(accelerator.device, dtype=weight_dtype)
    noise_scheduler = DDIMScheduler.from_pretrained(args.base_model_path + "scheduler")
    unet = UNet2DConditionModel.from_pretrained(args.base_model_path + "unet", use_safetensors=False).to(accelerator.device,
                                                                                  dtype=weight_dtype)

    init_adapter(unet, SkipAttnProcessor)
    attn_modules = get_trainable_module(unet, "attention")
    if args.resume_path is not None:
        unet.load_state_dict(torch.load(args.resume_path), strict=False)
        print("load model from", args.resume_path)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    for param in unet.parameters():
        param.requires_grad = False
    for blocks in [unet.down_blocks, unet.up_blocks]:
        for block in blocks:
            if hasattr(block, "attentions"):
                for block_module in block.attentions:
                    for transformer_block in block_module.transformer_blocks:
                        for param in transformer_block.attn1.parameters():
                            param.requires_grad = True
    for block_module in unet.mid_block.attentions:
        for transformer_block in block_module.transformer_blocks:
            for param in transformer_block.attn1.parameters():
                param.requires_grad = True

    params = filter(lambda p: p.requires_grad, unet.parameters())
    trainable_module = [name for name, param in unet.named_parameters() if param.requires_grad]
    print("trainable_module", len(trainable_module))
    vae.requires_grad_(False)
    unet.train()
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    train_dataset = HanscaDataset(args=args, is_test=False, is_paired=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_paired_dataset = HanscaDataset(args=args, is_test=True, is_paired=False)
    val_paired_dataloader = torch.utils.data.DataLoader(
        val_paired_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    unet, optimizer, train_dataloader, val_paired_dataloader = accelerator.prepare(unet, optimizer, train_dataloader, val_paired_dataloader)
    concat_dim = -2
    for epoch in range(args.start_epochs, args.max_epochs):
        if epoch % args.val_freq == 0:
            val(unet=unet, vae=vae, val_dataloader=val_paired_dataloader, noise_scheduler=noise_scheduler, epoch=epoch,
                accelerator=accelerator, args=args)
        loop = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in loop:
            image = batch['person'].to(accelerator.device)
            cloth = batch['cloth'].to(accelerator.device)
            mask = batch['mask'].to(accelerator.device)
            masked_image = image * (mask < 0.5)
            image_latent = compute_vae_encodings(image, vae)
            masked_latent = compute_vae_encodings(masked_image, vae)
            condition_latent = compute_vae_encodings(cloth, vae)
            mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
            del cloth, mask, image
            if random.random() < 0.1:
                masked_latent_concat = torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim)
            else:
                masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
            mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
            image_latent_concat = torch.cat([image_latent, condition_latent], dim=concat_dim)
            noise = torch.randn_like(image_latent_concat)
            bsz = image_latent_concat.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
            )
            noisy_latents = noise_scheduler.add_noise(image_latent_concat, noise, timesteps)
            inpainting_latent_model_input = torch.cat(
                [noisy_latents, mask_latent_concat, masked_latent_concat], dim=1)
            with torch.no_grad():
                unet.eval()
                noisy_pred = unet(inpainting_latent_model_input, timesteps, encoder_hidden_states=None).sample
                unet.train()
                error = noise - noisy_pred
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
                alphas_cumprod = alphas_cumprod[timesteps]
                sqrt_one_min_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
                noisy_latents_pred = noisy_latents + (sqrt_one_min_alphas_cumprod ** (args.p + 1)).view(-1, 1, 1,
                                                                                                        1) * error
                noisy_pred = noise + (sqrt_one_min_alphas_cumprod ** (args.p)).view(-1, 1, 1, 1) * error

            inpainting_latent_model_input = torch.cat(
                [noisy_latents_pred, mask_latent_concat, masked_latent_concat], dim=1)
            noise_rec = unet(inpainting_latent_model_input, timesteps, encoder_hidden_states=None).sample
            loss = F.mse_loss(noise_rec, noisy_pred, reduction="mean")
            avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            loop.set_description(f'Epoch [{epoch}/{args.max_epochs}]')
            loop.set_postfix(loss=loss.item())
        if epoch % args.save_epoch_freq == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwarped_unet = accelerator.unwrap_model(unet)
                state = {}
                for k, v in unwarped_unet.state_dict().items():
                    if "attn1" in k:
                        state[k] = v
                torch.save(state, args.save_path + f"/epoch_{epoch}_attn1.pth")
                print(f"Saved model at epoch {epoch}")

if __name__ == '__main__':
    main()