import torch
from diffusers import StableDiffusionXLPipeline, BitsAndBytesConfig, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from argparse import ArgumentParser, Namespace
from accelerate import Accelerator
from peft import LoraConfig
from utils.logger import setup_logger
import wandb
from bitsandbytes.optim import AdamW8bit
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# create global logger (maybe single instance will be better)
logger = setup_logger("sdxl_finetune", "logs/training.log")


def init_wandb(args: Namespace):
    wandb.init(
        project="sdxl-lora-finetuning",
        config=vars(args),
        name=f"sdxl-lora-rank{args.lora_rank}-bs{args.train_batch_size}-lr{args.learning_rate}",
    )
    logger.info("Initialized Weights & Biases logging.")


def train(args: Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Load model with 8-bit quantization
    logger.info(f"Loading model from {args.pretrained_model_name_or_path} with 8-bit quantization.")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision="fp16",
        torch_dtype=dtype,
        quantization_config=quantization_config,
    ).to(device)

    tokenizer1 = pipe.tokenizer
    tokenizer2 = pipe.tokenizer_2
    text_encoder1 = pipe.text_encoder
    text_encoder2 = pipe.text_encoder_2
    vae = pipe.vae
    unet: UNet2DConditionModel = pipe.unet
    noise_scheduler = pipe.scheduler

    # Freeze all model parameters
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Apply LoRA to UNet
    logger.info("Applying LoRA to UNet.")
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

    # Initialize 8-bit AdamW optimizer for LoRA parameters
    lora_params = [p for p in unet.parameters() if p.requires_grad]

    optimizer = AdamW8bit(
        lora_params,
        lr=args.learning_rate
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")

    # Data Paths and Prompts
    parser.add_argument("--instance_data_dir", type=str, required=True,
                        help="Path to the directory containing instance images.")
    parser.add_argument("--prior_data_dir", type=str, required=True,
                        help="Path to the directory containing prior preservation class images.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0,
                        help="Weight for the prior preservation loss.")
    parser.add_argument("--revision", type=str, default="fp16",
                        help="Model revision to load (e.g., 'fp16' or 'bf16').")

    # Training Parameters
    parser.add_argument("--output_dir", type=str, default="./lora_output_sdxl")
    parser.add_argument("--resolution", type=int, default=1024, help="The resolution for input images.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                        help="Total number of training steps to perform.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")

    # LoRA Parameters
    parser.add_argument("--lora_rank", type=int, default=32, help="The dimension of the LoRA matrices (rank).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="The alpha parameter for LoRA scaling.")
    parser.add_argument("--lora_name", type=str, default="my_sdxl_lora.safetensors",
                        help="File name for the saved LoRA weights.")

    # Optimizer and Scheduler
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for the AdamW optimizer.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of steps for the learning rate warmup.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
