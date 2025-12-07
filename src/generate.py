from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch
from peft import  LoraConfig
import argparse
import os
import yaml


def generate_images_with_lora_style(prompts, args: argparse.Namespace):
    """
    Generate images using a base SDXL model with LoRA weights,
    automatically appending a style keyword to each prompt.
    """
    # Choose device: prefer GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Set random seed for reproducible results
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # 1\. Load base SDXL pipeline (without LoRA, only base weights)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    )

    # Optionally disable NSFW safety checker to avoid filtered outputs
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    # 1\.5 Build the same LoRA structure on UNet as in training
    target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2"
    ]

    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        bias="none"
    )
    pipe.unet.add_adapter(lora_config, "lora_unet")

    # 2\. Load LoRA state\_dict and merge it into the pipeline UNet
    assert os.path.isfile(args.lora_path), f"LoRA weights file does not exist: {args.lora_path}"

    # Read LoRA weights from safetensors file (keys correspond to UNet LoRA layers)
    lora_state_dict = load_file(args.lora_path, device="cpu")

    with open("lora_state_dict_keys.yaml", "w") as f:
        yaml.dump(list(lora_state_dict.keys()), f)

    fixed_lora_state_dict = {}
    for k, v in lora_state_dict.items():
        # Skip non-LoRA parameters saved by peft
        if "lora_A" not in k and "lora_B" not in k:
            continue

        new_k = k

        # Only modify LoRA A/B weights: insert `.lora_unet` before `.weight`
        if new_k.endswith(".weight"):
            if "lora_A" in new_k:
                # ...lora_A.weight -> ...lora_A.lora_unet.weight
                new_k = new_k.replace("lora_A.weight", "lora_A.lora_unet.weight")
            elif "lora_B" in new_k:
                # ...lora_B.weight -> ...lora_B.lora_unet.weight
                new_k = new_k.replace("lora_B.weight", "lora_B.lora_unet.weight")

        fixed_lora_state_dict[new_k] = v

    # Load LoRA weights into UNet; only matching keys will be updated
    missing, unexpected = pipe.unet.load_state_dict(fixed_lora_state_dict, strict=False)
    print(f"LoRA loaded, missing_keys: {len(missing)}, unexpected_keys: {len(unexpected)}")

    # Move the entire pipeline to the selected device
    pipe.unet.eval()
    pipe.to(device)

    # 3\. Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Beginning image generation...")
    # 4\. Generate images for each prompt
    for i, prompt in enumerate(prompts):
        # Append style keyword to the original prompt if provided
        final_prompt = f"{prompt}, {args.style_keyword}" if args.style_keyword else prompt

        with torch.no_grad():
            image = pipe(final_prompt, generator=generator).images[0]

        save_path = output_path / f"generated_image_{i + 1}.png"
        image.save(save_path)
        print(f"Saved image for prompt '{final_prompt}' to {save_path}")

    print("Finished image generation.")


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    # Load config from YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace(**config)
    generate_images_with_lora_style(sample_prompts, args)
