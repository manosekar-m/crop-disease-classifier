import os
import numpy as np
from PIL import Image

classes = [
    "Healthy", "Bacterial_Blight", "Brown_Spot", "Blast_Fungus", 
    "Leaf_Scald", "Sheath_Rot", "False_Smut", "Tungro_Virus", 
    "Narrow_Brown_Spot", "Sheath_Blight"
]
splits = ["train", "val", "test"]

base_dir = "data/processed"

for split in splits:
    for cls in classes:
        dir_path = os.path.join(base_dir, split, cls)
        os.makedirs(dir_path, exist_ok=True)
        # Create a few dummy images per class per split
        num_images = 5 if split == "train" else 2
        for i in range(num_images):
            img_path = os.path.join(dir_path, f"img_{i}.jpg")
            # Create a random RGB image of 380x380
            data = np.random.randint(0, 255, (380, 380, 3), dtype=np.uint8)
            img = Image.fromarray(data, 'RGB')
            img.save(img_path)

print("Dummy data created successfully!")
