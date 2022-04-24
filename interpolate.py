#!/usr/bin/env python3

import torch
import pickle

with open('../models/gamma500/network-snapshot-010000.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema']# torch.nn.Module
z = torch.randn([1, G.z_dim])# latent codes
c = None                                # class labels (not used in this example)
img = G(z, c, force_fp32=True)                           # NCHW, float32, dynamic range [-1, +1]
