# server.py (Example)
import gradio as gr
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
import torch

# Load the Stable Diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Or any other model you like
).to(
    "cpu"
)  # Or "cpu" if you don't have a GPU


def generate_image(prompt):
    image = pipeline(prompt).images[0]
    return image


iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
)
iface.launch(server_name="0.0.0.0", server_port=8080)  # Or choose a different port
