#!/usr/bin/env python3

import gradio as gr

import numpy as np
import torch
import pickle
import PIL.Image
import types

from projector import project, imageio, _MODELS

from huggingface_hub import hf_hub_url, cached_download

# with open("../models/gamma500/network-snapshot-010000.pkl", "rb") as f:
# with open("../models/gamma400/network-snapshot-010600.pkl", "rb") as f:
# with open("../models/gamma400/network-snapshot-019600.pkl", "rb") as f:
with open(cached_download(hf_hub_url('ykilcher/apes', 'gamma500/network-snapshot-010000.pkl')), 'rb') as f:
    G = pickle.load(f)["G_ema"]  # torch.nn.Module

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    G = G.to(device)
else:
    _old_forward = G.forward

    def _new_forward(self, *args, **kwargs):
        kwargs["force_fp32"] = True
        return _old_forward(self, *args, **kwargs)

    G.forward = types.MethodType(_new_forward, G)

    _old_synthesis_forward = G.synthesis.forward

    def _new_synthesis_forward(self, *args, **kwargs):
        kwargs["force_fp32"] = True
        return _old_synthesis_forward(self, *args, **kwargs)

    G.synthesis.forward = types.MethodType(_new_synthesis_forward, G.synthesis)


def generate(
    target_image_upload,
    # target_image_webcam,
    num_steps,
    seed,
    learning_rate,
    model_name,
    normalize_for_clip,
    loss_type,
    regularize_noise_weight,
    initial_noise_factor,
):
    seed = round(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target_image = target_image_upload
    # if target_image is None:
        # target_image = target_image_webcam
    num_steps = round(num_steps)
    print(type(target_image))
    print(target_image.dtype)
    print(target_image.max())
    print(target_image.min())
    print(target_image.shape)
    target_pil = PIL.Image.fromarray(target_image).convert("RGB")
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    )
    target_pil = target_pil.resize(
        (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS
    )
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_image = torch.from_numpy(target_uint8.transpose([2, 0, 1])).to(device)
    projected_w_steps = project(
        G,
        target=target_image,
        num_steps=num_steps,
        device=device,
        verbose=True,
        initial_learning_rate=learning_rate,
        model_name=model_name,
        normalize_for_clip=normalize_for_clip,
        loss_type=loss_type,
        regularize_noise_weight=regularize_noise_weight,
        initial_noise_factor=initial_noise_factor,
    )
    with torch.no_grad():
        video = imageio.get_writer(f'proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        for w in projected_w_steps:
            synth_image = G.synthesis(w.to(device).unsqueeze(0), noise_mode="const")
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = (
                synth_image.permute(0, 2, 3, 1)
                .clamp(0, 255)
                .to(torch.uint8)[0]
                .cpu()
                .numpy()
            )
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()
    return synth_image, "proj.mp4"


iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.inputs.Image(source="upload", optional=True),
        # gr.inputs.Image(source="webcam", optional=True),
        gr.inputs.Number(default=250, label="steps"),
        gr.inputs.Number(default=69420, label="seed"),
        gr.inputs.Number(default=0.05, label="learning_rate"),
        gr.inputs.Dropdown(default='RN50', label="model_name", choices=['vgg16', *_MODELS.keys()]),
        gr.inputs.Checkbox(default=True, label="normalize_for_clip"),
        gr.inputs.Dropdown(
            default="l2", label="loss_type", choices=["l2", "l1", "cosine"]
        ),
        gr.inputs.Number(default=1e5, label="regularize_noise_weight"),
        gr.inputs.Number(default=0.05, label="initial_noise_factor"),
    ],
    outputs=["image", "video"],
)
iface.launch(inbrowser=True)
