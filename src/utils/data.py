import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from pathlib import Path


class CustomImageTextDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer1, tokenizer2, image_size: int = 1024):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.samples = self._load_samples()

        # Additional information
        self.origin_size = (image_size, image_size)
        self.target_size = (image_size, image_size)
        self.crop_coords = (0, 0)

    def _load_samples(self):
        samples = []
        for file in os.listdir(self.data_dir):
            file_path = Path(self.data_dir) / file
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                text_file_path = Path(self.data_dir) / (file_path.stem + "_generated.txt")
                if text_file_path.exists():
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    samples.append((str(file_path), text))
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
            return_tensors="pt"
        )

        tokens2 = self.tokenizer2(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        add_time_ids = list(self.origin_size + self.target_size + self.crop_coords)
        add_time_ids = torch.tensor(add_time_ids, dtype=torch.float32)

        return {
            "pixel_values": image,
            "input_ids_1": tokens1.input_ids.squeeze(),
            "attention_mask_1": tokens1.attention_mask.squeeze(),
            "input_ids_2": tokens2.input_ids.squeeze(),
            "attention_mask_2": tokens2.attention_mask.squeeze(),
            "add_time_ids": add_time_ids
        }


class CustomDataCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        input_ids_1 = torch.stack([item["input_ids_1"] for item in batch])
        attention_mask_1 = torch.stack([item["attention_mask_1"] for item in batch])
        input_ids_2 = torch.stack([item["input_ids_2"] for item in batch])
        attention_mask_2 = torch.stack([item["attention_mask_2"] for item in batch])
        add_time_ids = torch.stack([item["add_time_ids"] for item in batch])

        return {
            "pixel_values": pixel_values,
            "input_ids_1": input_ids_1,
            "attention_mask_1": attention_mask_1,
            "input_ids_2": input_ids_2,
            "attention_mask_2": attention_mask_2,
            "add_time_ids": add_time_ids
        }
