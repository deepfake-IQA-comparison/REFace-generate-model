import torch

ckpt = torch.load("/home/seclab/hdd3/REFace/checkpoints/last.ckpt", map_location="cpu", weights_only=False)

print(ckpt.keys())
