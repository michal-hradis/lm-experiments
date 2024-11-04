import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

while True:
    prompt = input("Enter a prompt: ")
    out = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        height=768,
        width=1360,
        num_inference_steps=50,
    ).images[0]
    out.save(prompt + ".png")