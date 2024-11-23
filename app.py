import torch
from diffusers import FluxPipeline

# Load the pipeline using default precision (FP32)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float32  # Default precision
).to("cuda")  # Ensure it runs on GPU

# Generate an image
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,  # Minimal guidance scale
    num_inference_steps=4,  # Few steps to reduce resource use
    max_sequence_length=128  # Short sequence for efficiency
).images[0]

# Save the generated image
image.save("flux-schnell.png")
