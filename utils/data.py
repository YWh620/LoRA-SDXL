import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
from torchvision import transforms


class CustomImageTextDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer1, tokenizer2, image_size: int = 512):
        self.data_dir = data_dir
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                base_name = filename[:-4]
                image_path = os.path.join(self.data_dir, base_name + ".png")
                text_path = os.path.join(self.data_dir, filename)
                if os.path.exists(image_path):
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    samples.append((image_path, text))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Tokenize text for both tokenizers
        tokens1 = self.tokenizer1(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        tokens2 = self.tokenizer2(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "image": image,
            "tokens1": tokens1,
            "tokens2": tokens2,
        }