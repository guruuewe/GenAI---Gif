from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
import os

def get_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda") 
    return pipe

def generate_frames(prompts, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    pipe = get_model()
    frames = []
    for i, prompt in enumerate(prompts):
        print(f"Generating frame {i + 1}/{len(prompts)}...")
        image = pipe(prompt).images[0]
        frame_path = f"{output_dir}/frame_{i:03d}.png"
        image.save(frame_path)  
        frames.append(frame_path)

    print("Frames saved in:", output_dir)
    return frames

def create_gif(frames, output_path="animation.gif", duration=300, loop=0):
    print(f"Creating GIF from {len(frames)} frames...")
    images = [Image.open(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop 
    )
    print(f"GIF saved at: {output_path}")

def main():
    prompts = []
    for i in range(4):
        prompt = input(f"Enter the prompt for frame {i + 1}: ")
        prompts.append(prompt)

    output_dir = "frames"
    frames = generate_frames(prompts, output_dir=output_dir)

    create_gif(frames, output_path="animation.gif", duration=300, loop=0)

if __name__ == "__main__":
    main()
