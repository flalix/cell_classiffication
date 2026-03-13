#!/usr/bin/env python3

from platform import python_version
print(python_version())

# echo $CONDA_DEFAULT_ENV

import numpy as np
import os, sys
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from natsort import natsorted


import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))


class _NoAutocast:
    def __init__(self, enabled=False): pass
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False

torch.cuda.amp.autocast = _NoAutocast   # global monkey-patch: autocast becomes a no-op

torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from cellpose import models, core, io, plot

print(io.logger_setup())

model = models.CellposeModel(gpu=True, model_type='cyto')

if hasattr(model, 'net'):
    # move to cuda then convert to float
    model.net = model.net.to('cuda').float()
    # sanity check: all params should be float32
    for n, p in list(model.net.named_parameters())[:5]:
        assert p.dtype == torch.float32, f"param {n} is {p.dtype}"
    print("Model parameters converted to:", next(model.net.parameters()).dtype)

net = getattr(model, 'net', None)
if net is None:
    raise RuntimeError("Could not find model.net — adjust this to your model object")
print(net)

# move to GPU and convert everything to float32
net.to('cuda')
for name, p in net.named_parameters(recurse=True):
    if p.dtype != torch.float32:
        p.data = p.data.float()
for name, buf in net.named_buffers(recurse=True):
    if buf.dtype != torch.float32:
        # some buffers might be empty tensors; guard against that
        if buf.numel() > 0:
            buf.data = buf.data.float()

# sanity check
dtypes = set(p.dtype for p in net.parameters())
if dtypes != {torch.float32}:
    print("Warning: not all params are float32:", dtypes)
else:
    print("All model parameters are float32")

print("\n-----------------\n")
print(net)

# --- targeted patch: ensure attention runs in FP32 (if SAM-based attention used) ---
try:
    import segment_anything.modeling.image_encoder as sa_enc
    orig_attn_forward = sa_enc.Attention.forward
    def attn_forward_force_fp32(self, x, *args, **kwargs):
        # convert input to float32 on the same device
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        # ensure weights/biases are float32 (redundant after step 1 but safe)
        with torch.cuda.amp.autocast(enabled=False):
            out = orig_attn_forward(self, x, *args, **kwargs)
        # if the original code returns bfloat16 or float16, convert outputs back to float32
        if isinstance(out, tuple):
            out = tuple(o.to(dtype=torch.float32) if isinstance(o, torch.Tensor) else o for o in out)
        elif isinstance(out, torch.Tensor):
            out = out.to(dtype=torch.float32)
        return out
    sa_enc.Attention.forward = attn_forward_force_fp32
    print("Patched segment_anything Attention.forward -> forces FP32")
except Exception as e:
    print("\n\n---------------------------------------------\n")
    print("Patch failed or segment_anything not present:", e)

    
is_laptop = False

if is_laptop:
    root_image = "../../colaboracoes/carlos_deOcesano"
else:
    root_image = "/media/flalix/d2f268d1-512d-499f-b3b3-6dad7d3fdd25/colaboracoes/deOcesano"


root_hcs = os.path.join(root_image, 'Plate1848')
hcs_folders = os.listdir(root_hcs)
print(">>> root_hcs", root_hcs)


root_1perc = os.path.join(root_hcs, '1% SFB')
ret = os.path.exists(root_1perc), root_1perc

print(f"1% SFB exists {ret}")

if not ret:
    exit(-1)

files = os.listdir(root_1perc)

i=0
filefig = os.path.join(root_1perc, files[i])
img = io.imread(filefig)

img2 = img.astype(np.float32) / 255.0

first_channel = '0' # @param ['None', 0, 1, 2, 3, 4, 5]
second_channel = '1' # @param ['None', 0, 1, 2, 3, 4, 5]
third_channel = '2' # @param ['None', 0, 1, 2, 3, 4, 5]

selected_channels = []
for i, c in enumerate([first_channel, second_channel, third_channel]):
  if c == 'None':
    continue
  if int(c) > img2.shape[-1]:
    assert False, 'invalid channel index, must have index greater or equal to the number of channels'
  if c != 'None':
    selected_channels.append(int(c))


img_selected_channels = np.zeros_like(img2, np.float32)
img_selected_channels[:, :, :len(selected_channels)] = img2[:, :, selected_channels]


flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

# --- now run inference in a no_grad block with smaller batch size for safety ---
with torch.no_grad():
    masks, flows, styles = model.eval(
        img_selected_channels,
        batch_size=8,  # reduce 32 to 8, while debugging
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize}
    )


fig = plt.figure(figsize=(12,5))
plot.show_segmentation(fig, img_selected_channels, masks, flows[0])
plt.tight_layout()
plt.show()


