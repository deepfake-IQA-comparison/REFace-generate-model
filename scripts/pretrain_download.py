from transformers import AutoModel
import os

model_name = "Sanoojan/REFace/last.ckpt"
checkpoint_dir = "./model/checkpoints"

model = AutoModel.from_pretrained(model_name)
model.save_pretrained(checkpoint_dir)