from collections import Counter
from pathlib import Path

def load_class_names(txt_path):
    id_to_name = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                synset_id, name = parts
                id_to_name[synset_id] = name
    return id_to_name

def count_images_per_class(root_folder):
    class_counts = Counter()
    for split in ['train', 'val']:
        split_path = Path(root_folder) / split
        for class_folder in split_path.iterdir():
            if class_folder.is_dir():
                count = sum(1 for file in class_folder.glob("*") if file.is_file())
                class_counts[class_folder.name] += count
    return class_counts

dataset_root = "/home/sebastien/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini"
class_map_txt = "words.txt"

id_to_name = load_class_names(class_map_txt)
counts = count_images_per_class(dataset_root)

named_counts = {id_to_name.get(cls_id, cls_id): count for cls_id, count in counts.items()}

import pandas as pd
df = pd.DataFrame(named_counts.items(), columns=["Class Name", "Image Count"])
df = df.sort_values("Image Count", ascending=False)

output_csv_path = "imagenet_class_distribution.csv"
df.to_csv(output_csv_path, index=False)
