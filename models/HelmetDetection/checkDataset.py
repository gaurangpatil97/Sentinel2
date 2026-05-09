import os

dataset_path = r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\dataset"

for split in ["train", "valid"]:
    images = len(os.listdir(os.path.join(dataset_path, split, "images")))
    labels = len(os.listdir(os.path.join(dataset_path, split, "labels")))
    print(f"{split}: {images} images, {labels} labels")

print("\ndata.yaml contents:")
with open(os.path.join(dataset_path, "data.yaml")) as f:
    print(f.read())