import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, PipelineQuantizationConfig, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from argparse import ArgumentParser, Namespace
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig
from torch.utils.data import DataLoader

from utils.logger import setup_logger
import wandb
from bitsandbytes.optim import AdamW8bit
from utils.data import CustomImageTextDataset, CustomDataCollator
import os
import yaml


def init_wandb(args: Namespace):
    wandb.init(
        project="sdxl-lora-finetuning",
        config=vars(args),
        name=f"sdxl-lora-rank{args.lora_rank}-bs{args.train_batch_size}-lr{args.learning_rate}",
        dir="wandb_logs"
    )


def compute_sdxl_unet_loss(unet, noise_scheduler, text_encoder1, text_encoder2, batch, device, dtype):
    # Get the latents and add noise
    latents = batch["latents"].to(device, dtype=dtype)
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get text embeddings from both text encoders
    input_ids_1 = batch["input_ids_1"]
    attention_mask_1 = batch["attention_mask_1"]
    input_ids_2 = batch["input_ids_2"]
    attention_mask_2 = batch["attention_mask_2"]

    # Dual text encoding
    with torch.no_grad():
        text_embeddings1 = text_encoder1(input_ids_1, attention_mask=attention_mask_1)
        text_embeddings2 = text_encoder2(input_ids_2, attention_mask=attention_mask_2)
        encoder_hidden_state_1 = text_embeddings1.last_hidden_state
        encoder_hidden_state_2 = text_embeddings2.last_hidden_state
        pooled_text_embeds = text_embeddings2.pooler_output

    encoder_hidden_state = torch.cat([encoder_hidden_state_1, encoder_hidden_state_2], dim=-1)

    # Predict the noise residual
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_state,
        added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": batch["add_time_ids"]}
    ).sample

    # Compute MSE loss
    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
    return loss


def train(args: Namespace):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if dtype == torch.bfloat16 else "fp16"
    )

    # create global logger for each process (maybe single instance will be better)
    logger = setup_logger("sdxl_finetune",
                          f"logs/training_device{accelerator.local_process_index}.log",
                          accelerator.is_main_process)

    set_seed(42)
    logger.info(f"Using device: {accelerator.device}, dtype: {dtype}")

    # Load pre-trained Stable Diffusion XL model
    logger.info(f"Loading model from {args.pretrained_model_name_or_path} with 8-bit quantization")
    quantization_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True}
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        quantization_config=quantization_config
    )

    tokenizer1 = pipe.tokenizer
    tokenizer2 = pipe.tokenizer_2
    text_encoder1 = pipe.text_encoder
    text_encoder2 = pipe.text_encoder_2
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = pipe.scheduler

    # Freeze all model parameters
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="UNET_KD"
    )
    unet.add_adapter(lora_config, "lora_unet")

    # Print trainable parameters
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(
        f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

    # Initialize datasets and dataloaders
    cyber_dataset = CustomImageTextDataset(args.instance_data_dir, tokenizer1, tokenizer2, args.resolution)
    prior_dataset = CustomImageTextDataset(args.prior_data_dir, tokenizer1, tokenizer2, args.resolution)

    cyber_dataloader = DataLoader(cyber_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=CustomDataCollator())
    prior_dataloader = DataLoader(prior_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=CustomDataCollator())

    logger.info(
        f"Total batch size: {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # Initialize 8-bit AdamW optimizer for LoRA parameters
    lora_params = [p for p in unet.parameters() if p.requires_grad]

    optimizer = AdamW8bit(
        lora_params,
        lr=args.learning_rate
    )

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    unet, optimizer, cyber_dataloader, prior_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, cyber_dataloader, prior_dataloader, lr_scheduler
    )

    vae = vae.to(accelerator.device, dtype=dtype)
    text_encoder1 = text_encoder1.to(accelerator.device, dtype=dtype)
    text_encoder2 = text_encoder2.to(accelerator.device, dtype=dtype)

    prior_data_iter = iter(prior_dataloader)

    logger.info("Model and env initialization complete. Starting training...")
    global_step = 0
    unet.train()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        init_wandb(args)

    while global_step < args.max_train_steps:
        for step, batch in enumerate(cyber_dataloader):
            # Get prior batch
            try:
                prior_batch = next(prior_data_iter)
            except StopIteration:
                prior_data_iter = iter(accelerator.prepare(prior_dataloader))
                prior_batch = next(prior_data_iter)

            # VAE encode images
            with torch.no_grad():
                cyber_latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                prior_latents = vae.encode(prior_batch["pixel_values"]).latent_dist.sample()
                batch["latents"] = cyber_latents * vae.config.scaling_factor
                prior_batch["latents"] = prior_latents * vae.config.scaling_factor

            with accelerator.accumulate(unet):
                cyber_loss = compute_sdxl_unet_loss(
                    unet, noise_scheduler, text_encoder1, text_encoder2, batch, accelerator.device, dtype)
                prior_loss = compute_sdxl_unet_loss(
                    unet, noise_scheduler, text_encoder1, text_encoder2, prior_batch, accelerator.device, dtype)
                total_loss = cyber_loss + args.prior_loss_weight * prior_loss

                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

            if global_step > 0 and global_step % args.logging_steps == 0 and accelerator.is_main_process:
                logger.info(
                    f"Step {global_step}: cyber_loss={cyber_loss.item():.4f}, prior_loss={prior_loss.item():.4f}, "
                    f"total_loss={total_loss.item():.4f}, lr={lr_scheduler.get_last_lr()[0]:.6f}"
                )
                wandb.log({
                    "train/cyber_loss": cyber_loss.item(),
                    "train/prior_loss": prior_loss.item(),
                    "train/total_loss": total_loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step": global_step
                }, step=global_step)

            if global_step > 0 and global_step % args.save_steps == 0 and accelerator.is_main_process:
                unet_unwrapped: UNet2DConditionModel = accelerator.unwrap_model(unet)
                unet_unwrapped.save_pretrained(
                    save_directory=os.path.join(args.output_dir, f"unet_lora_step{global_step}"),
                    safe_serialization=True,
                    is_main_process=accelerator.is_main_process
                )

                logger.info(f"Saved LoRA weights at step {global_step}")

            if global_step >= args.max_train_steps:
                break

    # Final save
    if accelerator.is_main_process:
        unet_unwrapped: UNet2DConditionModel = accelerator.unwrap_model(unet)
        unet_unwrapped.save_pretrained(
            save_directory=os.path.join(args.output_dir, "unet_lora_final"),
            safe_serialization=True,
            is_main_process=accelerator.is_main_process
        )
        logger.info("Training complete. Final LoRA weights saved.")

    logger.info(f"Training complete.")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    # Load config from YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    args = Namespace(**config)
    # Start training
    train(args)


if __name__ == "__main__":
    main()
