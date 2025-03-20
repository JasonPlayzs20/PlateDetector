import os

image_folder = "train/images/train"  # Update this path
output_file = "train/train.txt"

with open(output_file, "w") as f:
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            f.write(os.path.abspath(os.path.join(image_folder, filename)) + "\n")

print("✅ train.txt created successfully!")