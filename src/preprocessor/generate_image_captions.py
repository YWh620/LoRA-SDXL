from openai import OpenAI
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import base64


def image_to_base64_data_url(image_path):
    img = Image.open(image_path)
    mime_type = Image.MIME[img.format]
    img.close()
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded_string}"
    return data_url


def generate_image_captions(api_key, local_dir,
                            prompt="Generate a detailed and descriptive caption for the following image."):
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print(f"Begin generating image captions for images in {local_dir}")
    files = os.listdir(local_dir)
    files = [file for file in files if Path(file).suffix == ".jpg"]
    for file in tqdm(files, desc="Generating image captions..."):
        file_path = Path(os.path.join(local_dir, file))
        if file_path.suffix != ".jpg":
            continue
        img_url = image_to_base64_data_url(file_path)
        completion = client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        }
                    }
                ]
            }]
        )
        caption = completion.choices[0].message.content.strip()
        with open(f"{local_dir}/{file_path.stem}_generated.txt", "w") as f:
            f.write(caption)
    print(f"End generating image captions for images in {local_dir}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--api_key", type=str, required=True, help="Qwen llm API Key")
    config = args.parse_args()
    # generate captions for cyberpunk images
    cyberpunk_dir = "../../data/cyberpunk_cities_preprocessed"
    cyber_prompt = ("Generate a concise (20–30 words) description for a cyberpunk city scene. "
                    "Include the style tag cyberpunk city and naturally mention several of these elements: neon glows, "
                    "holographic projections, slick wet asphalt, brutalist mega-structures, flying vehicles, "
                    "data streams, electric rain, grungy alley, or atmospheric haze. "
                    "Output only the caption.")
    generate_image_captions(config.api_key, cyberpunk_dir, prompt=cyber_prompt)

    # generate captions for ppl images
    ppl_dir = "../../data/general_cities_preprocessed"
    ppl_prompt = ("Generate a concise (15–25 words) caption for a realistic urban scene. "
                  "Include several key elements such as skyscrapers, glass facades, busy streets, street traffic, "
                  "pedestrians, crosswalks, street lamps, office buildings, or city park. Focus on neutral, "
                  "descriptive nouns and natural sentence flow. Avoid any adjectives related to mood, lighting, style, "
                  "or quality. "
                  "Output only the caption.")

    generate_image_captions(config.api_key, ppl_dir, prompt=ppl_prompt)


if __name__ == "__main__":
    main()
