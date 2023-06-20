import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks
from modules import shared
from modules.shared import opts
from modules import extensions
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image
import os
from typing import Callable, Any
import json
import fnmatch
import torch
import time

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Txt/Img to 3D Model"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def generate(mode, batch_size, prompt, use_karras, karras_steps, init_image, clip_denoised, use_fp16, guidance_scale,
             s_churn):
    print("mode:" + mode)
    print("clip_denoised:" + str(clip_denoised))
    print("use_fp16:" + str(use_fp16))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xm = load_model('transmitter', device=device)

    model_name = 'image300M' if mode == "img" else 'text300M'

    model = load_model(model_name, device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    model_kwargs = dict(texts=[prompt] * batch_size)

    if mode == "img":
        model_kwargs = dict(images=[init_image] * batch_size)

    print("loaded model:" + model_name)

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=model_kwargs,
        progress=True,
        clip_denoised=clip_denoised,
        use_fp16=use_fp16,
        use_karras=use_karras,
        karras_steps=karras_steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=s_churn,
    )

    print("sample_latents done")

    current_path = os.path.abspath(__file__)

    parent_path = os.path.dirname(current_path)

    parent_path = os.path.dirname(parent_path)

    output_dir = os.path.join(parent_path, 'outputs')

    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time())

    try:
        output_format = opts.txtimg_to_3d_model_output_format
    except:
        output_format = 'obj'

    for i, latent in enumerate(latents):
        output_file_path = os.path.join(output_dir, f'mesh_{timestamp}_{i}.{output_format}')

        t = decode_latent_mesh(xm, latent).tri_mesh()

        if output_format == 'obj':
            with open(output_file_path, 'w') as f:
                t.write_obj(f)
        else:
            with open(output_file_path, 'wb') as f:
                t.write_ply(f)


    print("output mesh done")

    return output_file_path


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as shap_e:
        with gr.Row():
            with gr.Column():
                mode = gr.Dropdown(["txt", "img"], label="Mode", value="img")
                prompt_txt = gr.Textbox(label="Prompt", lines=2)
                init_image = gr.Image(type="pil")
                batch_size_slider = gr.Slider(minimum=1, maximum=10, default=2, value=2, step=1, label="Batch Size",
                                              interactive=True)
                use_karras = gr.Checkbox(label="Use Karras", value=True)
                karras_steps_slider = gr.Slider(minimum=1, maximum=100, default=64, value=64, step=1,
                                                label="Karras Steps",
                                                interactive=True)
                clip_denoised = gr.Checkbox(label="Clip Denoised", value=True)
                use_fp16 = gr.Checkbox(label="Use fp16", value=True)
                guidance_scale_slider = gr.Slider(minimum=1, maximum=20, default=3, value=3, step=1,
                                                  label="Guidance Scale",
                                                  interactive=True)
                s_churn_slider = gr.Slider(minimum=0, maximum=5, default=0, value=0, step=1,
                                           label="S Churn",
                                           interactive=True)
                btn = gr.Button(value="Submit")
            with gr.Column():
                output1 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model")

        btn.click(fn=generate,
                  inputs=[mode, batch_size_slider, prompt_txt, use_karras, karras_steps_slider, init_image,
                          clip_denoised, use_fp16, guidance_scale_slider, s_churn_slider],
                  outputs=output1)

    return [(shap_e, "Txt/Img to 3D Model", "txt_img_to_3d_model")]


def on_ui_settings():
    section = ('txtimg_to_3d_model', "Txt/Img To 3d Model")
    shared.opts.add_option("txtimg_to_3d_model_output_format", shared.OptionInfo(
        "obj", "Output format (Only obj format can preview on the page)", gr.Radio, {"choices": ['obj', 'ply']}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
