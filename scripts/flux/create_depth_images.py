from transformers import pipeline
from PIL import Image
import requests
import os
import argparse

parser = argparse.ArgumentParser(description="Generate depth images from input images.")
parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input images.")
parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save depth images.")
args = parser.parse_args()

output_dir = args.output_folder
# Create output directory if it doesn't exist

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device=0)

# load images from the folder
image_folder = args.input_folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
images = [Image.open(image_file).convert("RGB") for image_file in image_files]

# process images
outputs = pipe(images)

# save outputs
for i, output in enumerate(outputs):
    output_depth = output["depth"]
    image_name = image_files[i].split("/")[-1].split(".")[0]
    print(f"Processing {image_name}")
    output_depth.save(f"{output_dir}/depth_{image_name}.png")
    print(f"Depth image saved for image {image_name}.png")