import requests
import json
import os
import argparse
from tqdm import tqdm

data_dir = os.path.join(os.path.dirname(__file__), "../../data")


def fetch_ppl_images(access_key):
    total_num = 0
    page = 1
    per_page = 30
    ppl_images = []
    while total_num < 540:
        url = f"https://api.unsplash.com/collections/83895897/photos?page={page}&per_page={per_page}"
        headers = {"Authorization": f"Client-ID {access_key}", "Accept-Version": "v1",
                   "Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching images: {response.status_code}")
            break
        results = response.json()
        results = [{"id": img["id"], "url": img["urls"]["regular"], 'origin_caption': img['alt_description']} for img in
                   results]
        ppl_images.extend(results)
        total_num += len(results)
        page += 1

    print(f"Fetched {len(ppl_images)} PPL images from Unsplash API")

    ppl_data_dir = os.path.join(data_dir, "general_cities")

    os.makedirs(ppl_data_dir, exist_ok=True)
    with open(os.path.join(ppl_data_dir, "ppl_images.json"), "w") as f:
        json.dump(ppl_images, f, indent=4)

    # download images
    print("Begin downloading PPL images...")
    for i, img in enumerate(tqdm(ppl_images, desc="Downloading PPL images...")):
        img_data = requests.get(img["url"]).content
        with open(os.path.join(ppl_data_dir, f"{i + 1:03d}.jpg"), "wb") as handler:
            handler.write(img_data)
        with open(os.path.join(ppl_data_dir, f"{i + 1:03d}.txt"), "w") as f:
            f.write(img['origin_caption'] if img['origin_caption'] else "No description available.")

    print("PPL images downloaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--access_key", type=str, required=True, help="Unsplash API Access Key")
    config = parser.parse_args()
    fetch_ppl_images(config.access_key)
