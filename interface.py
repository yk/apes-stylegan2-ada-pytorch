#!/usr/bin/env python3

import gradio as gr

import numpy as np
import torch
import pickle
import types

from huggingface_hub import hf_hub_url, cached_download

# with open('../models/gamma500/network-snapshot-010000.pkl', 'rb') as f:
with open(cached_download(hf_hub_url('ykilcher/apes', 'gamma500/network-snapshot-010000.pkl')), 'rb') as f:
    G = pickle.load(f)['G_ema']# torch.nn.Module

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


def generate(num_images, interpolate):
    if interpolate:
        z1 = torch.randn([1, G.z_dim])# latent codes
        z2 = torch.randn([1, G.z_dim])# latent codes
        zs = torch.cat([z1 + (z2 - z1) * i / (num_images-1) for i in range(num_images)], 0)
    else:
        zs = torch.randn([num_images, G.z_dim])# latent codes
    with torch.no_grad():
        zs = zs.to(device)
        img = G(zs, None, force_fp32=True, noise_mode='const') 
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img.cpu().numpy()

def greet(num_images, interpolate):
    img = generate(round(num_images), interpolate)
    imgs = list(img)
    if len(imgs) == 1:
        return imgs[0]
    grid_len = int(np.ceil(np.sqrt(len(imgs)))) * 2
    grid_height = int(np.ceil(len(imgs) / grid_len))
    grid = np.zeros((grid_height * imgs[0].shape[0], grid_len * imgs[0].shape[1], 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        y = (i // grid_len) * img.shape[0]
        x = (i % grid_len) * img.shape[1]
        grid[y:y+img.shape[0], x:x+img.shape[1], :] = img
    return grid


iface = gr.Interface(fn=greet, inputs=[
    gr.inputs.Number(default=1, label="Num Images"),
    gr.inputs.Checkbox(default=False, label="Interpolate")
    ], outputs="image")
iface.launch()
