from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch
import argparse
import os


def generate_images_with_lora_style(
        base_model: str,
        lora_path: str,  # path to LoRA weights file, e.g. ./lora_output_sdxl/lora_unet_final/lora_unet.safetensors
        prompts: list[str],
        output_dir: str,
        style_keyword: str = "cyberpunk city",
        seed: int = 42):
    """
    Generate images using a base SDXL model with LoRA weights,
    automatically appending a style keyword to each prompt.
    """
    # Choose device: prefer GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Set random seed for reproducible results
    generator = torch.Generator(device=device).manual_seed(seed)

    # 1\. Load base SDXL pipeline (without LoRA, only base weights)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )

    # Optionally disable NSFW safety checker to avoid filtered outputs
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    # 2\. Load LoRA state\_dict and merge it into the pipeline UNet
    assert os.path.isfile(lora_path), f"LoRA weights file does not exist: {lora_path}"

    # Read LoRA weights from safetensors file (keys correspond to UNet LoRA layers)
    lora_state_dict = load_file(lora_path, device="cpu")

    # Load LoRA weights into UNet; only matching keys will be updated
    missing, unexpected = pipe.unet.load_state_dict(lora_state_dict, strict=False)
    print(f"LoRA loaded, missing_keys: {len(missing)}, unexpected_keys: {len(unexpected)}")

    # Move the entire pipeline to the selected device
    pipe.to(device)
    pipe.eval()

    # 3\. Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 4\. Generate images for each prompt
    for i, prompt in enumerate(prompts):
        # Append style keyword to the original prompt if provided
        final_prompt = f"{prompt}, {style_keyword}" if style_keyword else prompt

        with torch.no_grad():
            image = pipe(final_prompt, generator=generator).images[0]

        save_path = output_path / f"generated_image_{i + 1}.png"
        image.save(save_path)
        print(f"Saved image for prompt '{final_prompt}' to {save_path}")


if __name__ == "__main__":
    # Example prompt list
    sample_prompts = [
        "A bustling city street with towering skyscrapers, neon signs, and pedestrians crossing under street lamps",
        "Crowded downtown avenue with glass office buildings, street traffic, people walking on sidewalks, and digital billboards",
        "Futuristic cityscape featuring highways, monorails above, tall skyscrapers, trams, and pedestrians on the streets",
        "Urban intersection with glass skyscrapers, busy streets, street vendors, and metro entrances surrounded by pedestrians",
        "Skyline view of a cyber city with high-rise buildings, bridges connecting them, street traffic, and people walking",
        "Industrial district with factories, smoke stacks, high-rise buildings, traffic lights, and pedestrians on busy streets",
        "City plaza surrounded by office towers, pedestrian zones, street lamps, outdoor cafes, and people strolling",
        "Futuristic market street filled with neon signs, crowded sidewalks, hover vehicles above, and street vendors selling goods",
        "Urban park surrounded by skyscrapers, walking paths, street lamps, people jogging, and bicycles parked near office buildings",
        "High-tech district with glass facades, elevated trains, crosswalks crowded with pedestrians, and digital billboards on buildings",
    ]

    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion XL and LoRA style.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base Stable Diffusion XL model name or local path.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        # This must match the final save path in train.py
        default="./lora_output_sdxl/lora_unet_final/lora_unet.safetensors",
        help="Path to LoRA state_dict weights file (.safetensors).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    generate_images_with_lora_style(
        base_model=args.base_model,
        lora_path=args.lora_path,
        prompts=sample_prompts,
        output_dir=args.output_dir,
        style_keyword="cyberpunk city",
        seed=args.seed,
    )
