import torch

pretrained_model_path='../../hdd3/REFace/pretrained_models/sd-v1-4.ckpt'
ckpt_file=torch.load(pretrained_model_path, map_location='cpu', weights_only = False)
zero_data=torch.zeros(320,5,3,3)
new_weight=torch.cat((ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'],zero_data),dim=1)
ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight']=new_weight
torch.save(ckpt_file,"../../hdd3/REFace/pretrained_models/sd-v1-4-modified-9channel.ckpt")