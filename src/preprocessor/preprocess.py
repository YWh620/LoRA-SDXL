from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
import shutil


# noinspection PyTypeChecker
def main():
    # first process the cyberpunk images
    cyberpunk_dir = "../../data/cyberpunk_cities"
    cyberpunk_preprocessed_dir = "../../data/cyberpunk_cities_preprocessed"
    shutil.rmtree(cyberpunk_preprocessed_dir, ignore_errors=True)
    os.mkdir(cyberpunk_preprocessed_dir)
    files = os.listdir(cyberpunk_dir)
    files = [file for file in files if Path(file).suffix == ".webp"]
    i = 0
    for file in tqdm(files, desc="Processing cyberpunk images..."):
        file = Path(os.path.join(cyberpunk_dir, file))
        if file.suffix != ".webp":
            continue
        img = Image.open(file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        save_path = os.path.join(cyberpunk_preprocessed_dir, f"{i + 1:03d}.jpg")
        img.save(save_path, format="JPEG")
        desc = file.stem
        with open(f"{cyberpunk_preprocessed_dir}/{i + 1:03d}.txt", "w") as f:
            bracket_pos = desc.find('(')
            if bracket_pos != -1:
                desc = desc[:bracket_pos].strip()
            f.write(desc)
        i += 1

    # then process the ppl images, only need to resize
    ppl_dir = "../../data/general_cities"
    ppl_preprocessed_dir = "../../data/general_cities_preprocessed"
    shutil.rmtree(ppl_preprocessed_dir, ignore_errors=True)
    os.mkdir(ppl_preprocessed_dir)
    files = os.listdir(ppl_dir)
    files = [file for file in files if Path(file).suffix == ".jpg"]
    i = 0
    for file in tqdm(files, desc="Processing PPL images..."):
        file = Path(os.path.join(ppl_dir, file))
        if file.suffix != ".jpg":
            continue
        img = Image.open(file)
        save_path = os.path.join(ppl_preprocessed_dir, f"{i + 1:03d}.jpg")
        img.save(save_path, format="JPEG")
        # copy the description file
        desc_src_path = os.path.join(ppl_dir, f"{file.stem}.txt")
        desc_dst_path = os.path.join(ppl_preprocessed_dir, f"{file.stem}.txt")
        with open(desc_src_path, "r") as f_src, open(desc_dst_path, "w") as f_dst:
            f_dst.write(f_src.read())
        i += 1


if __name__ == "__main__":
    main()
