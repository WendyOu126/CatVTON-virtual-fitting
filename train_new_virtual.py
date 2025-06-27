import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 配置4张RTX 3090 GPU
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from model.myPipeline import CatVTONPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import compute_vae_encodings, numpy_to_pil
import random
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs as DDPK
import json

# 转换张量为PIL图像（与原代码一致）
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

# 修改HanscaDataset以适配三元组数据集（无需掩码）
class HanscaDataset(Dataset):
    def __init__(self, args, is_test=False, is_paired=True):
        self.args = args
        self.data_root_path = args.data_root_path
        self.is_test = is_test
        self.is_paired = is_paired
        assert is_test or is_paired
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.data = self.load_data()

    def load_data(self):
        pair_txt = os.path.join(self.data_root_path, 'test_unpairs.json' if self.is_test else 'train_pairs.json')
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            json_data = json.load(f)
        
        data = []
        self.data_root_path = os.path.join(self.data_root_path, "test" if self.is_test else "train")
        
        # 解析三元组数据：clothes、models、composites
        for item in json_data:
            cloth_img = item['clothes'][0]  # 取第一个服装图像
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
        person, cloth, composite = [Image.open(data[key]) for key in ['person', 'cloth', 'composite']]
        return {
            'index': idx,
            'person_name': data['person_name'],
            'cloth_name': data['cloth_name'],
            'composite_name': data['composite_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'composite': self.vae_processor.preprocess(composite, self.args.height, self.args.width)[0]
        }

# 参数解析（适配4张GPU，默认fp16）
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
    parser.add_argument("--save_path", type=str, default="hansca_ckpts_nomask")
    parser.add_argument("--output_dir", type=str, default="hansca_output_nomask")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--start_epochs", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_epoch_freq", type=int, default=5)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--vae_path", type=str, default="/CatVTON/sd-vae-ft-mse")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", default=True)
    parser.add_argument("--use_tf32", default=True)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--concat_eval_results", default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

# 验证函数（适配三元组数据，移除掩码相关逻辑）
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
            results = pipeline(
                person_images,
                cloth_images,
                None,  # 无掩码输入
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            )
            if args.concat_eval_results:
                person_images = to_pil_image(person_images)
                cloth_images = to_pil_image(cloth_images)
                results = to_pil_image(results)
            for j, result in enumerate(results):
                person_name = batch['person_name'][j].split('.')[0]
                cloth_name = batch['cloth_name'][j].split('.')[0]
                output_path = os.path.join(output_dir, f"{person_name}_{cloth_name}.jpg")
                if args.concat_eval_results:
                    w, h = result.size
                    concated_result = Image.new('RGB', (w * 3, h))
                    concated_result.paste(person_images[j], (0, 0))
                    concated_result.paste(cloth_images[j], (w, 0))
                    concated_result.paste(result, (w * 2, 0))
                    result = concated_result
                result.save(output_path)
    del pipeline

# 主函数（修改训练逻辑以适配无需掩码）
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

    # 调整U-Net输入通道（移除掩码通道）
    unet.conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, kernel_size=3, padding=1).to(accelerator.device, dtype=weight_dtype)
    
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

    # 仅训练自注意力模块
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
            target = batch['composite'].to(accelerator.device)  # 使用合成试衣图像作为目标
            image_latent = compute_vae_encodings(image, vae)
            condition_latent = compute_vae_encodings(cloth, vae)
            target_latent = compute_vae_encodings(target, vae)
            del cloth, image, target
            if random.random() < 0.1:
                latent_concat = torch.cat([image_latent, torch.zeros_like(condition_latent)], dim=concat_dim)
            else:
                latent_concat = torch.cat([image_latent, condition_latent], dim=concat_dim)
            noise = torch.randn_like(target_latent)
            bsz = target_latent.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
            )
            noisy_latents = noise_scheduler.add_noise(target_latent, noise, timesteps)
            model_input = torch.cat([noisy_latents, latent_concat], dim=1)
            with torch.no_grad():
                unet.eval()
                noisy_pred = unet(model_input, timesteps, encoder_hidden_states=None).sample
                unet.train()
                error = noise - noisy_pred
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
                alphas_cumprod = alphas_cumprod[timesteps]
                sqrt_one_min_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
                noisy_latents_pred = noisy_latents + (sqrt_one_min_alphas_cumprod ** (args.p + 1)).view(-1, 1, 1, 1) * error
                noisy_pred = noise + (sqrt_one_min_alphas_cumprod ** (args.p)).view(-1, 1, 1, 1) * error

            model_input = torch.cat([noisy_latents_pred, latent_concat], dim=1)
            noise_rec = unet(model_input, timesteps, encoder_hidden_states=None).sample
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