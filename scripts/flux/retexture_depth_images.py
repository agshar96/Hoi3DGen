import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
import os
import argparse

parser = argparse.ArgumentParser(description="Generate images from depth maps using FluxControlPipeline.")
parser.add_argument("--depth_dir", type=str, required=True, help="Directory containing depth images.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts.")
parser.add_argument("--num_views", type=int, default=3, help="Number of views to process.")
args = parser.parse_args()

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to("cuda")

depth_dir = args.depth_dir
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

prompt_file = args.prompt_file


with open(prompt_file, "r") as file:
    prompts = file.readlines()
    # Read the prompts and add "White Background" to each prompt
    prompts = [prompt.strip() + ". White Background, PhotoRealistic, High Detail, Realistic Looking." for prompt in prompts]

for i, prompt in enumerate(prompts):
    # if i not in list_to_process:
    #     continue
    for j in range(args.num_views):
        output_path = f"{output_dir}/output_{i*args.num_views + j}.png"
        if os.path.exists(output_path):
            print(f"Image for prompt {i*args.num_views + j} already exists. Skipping...")
            continue

        print(f"Processing prompt {i*args.num_views + j + 1}/{len(prompts) * args.num_views}: {prompt}")
        image_path = f"{depth_dir}/depth_{i*args.num_views + j}.png"
        
        # Assuming the depth images are saved in the depth_outputs directory
        control_image = load_image(image_path).convert("RGB")

        result = pipe(
            prompt=prompt,
            control_image=control_image,
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=10.0,
            generator=torch.Generator().manual_seed(42),
        ).images[0]

        result.save(f"{output_dir}/output_{i*args.num_views + j}.png")
        print(f"Image generated and saved for prompt: {prompt}")

