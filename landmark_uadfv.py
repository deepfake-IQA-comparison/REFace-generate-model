import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from argparse import ArgumentParser
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps

landmarks = np.load("/home/seclab/hdd1/code/DeepfakeBench/datasets/UADFV/real/landmarks/0000/000.npy")
#print(landmarks)

parser = ArgumentParser()
parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
parser.add_argument('--faceParsing_ckpt', type=str, default="/home/seclab/hdd3/REFace/Other_dependencies/face_parsing/79999_iter.pth")  
		
parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file, '
                                                                    'this option is valid when --faceParsing_ckpt=segenext')
parser.add_argument('--image_root', type=str, default="/home/seclab/hdd1/code/DeepfakeBench/datasets/UADFV/real/frames")  
parser.add_argument('--save_vis', action='store_true')
parser.add_argument('--seg12', action='store_true')

args = parser.parse_args()

faceParsing_model = init_faceParsing_pretrained_model(args.faceParser_name, args.faceParsing_ckpt, args.segnext_config)
file_path = '/home/seclab/hdd1/code/DeepfakeBench/datasets/UADFV/real/frames/0000/009.png'
img = np.array(Image.open(file_path).convert("RGB"))
seg = faceParsing_demo(faceParsing_model, img)
print(np.unique(seg))
print(len(np.unique(seg)))
save_dir = "/home/seclab/hdd1/code/DeepfakeBench/datasets/UADFV/real/0000"
os.makedirs(save_dir, exist_ok=True)
Image.fromarray(seg.astype(np.uint8)).save("mask.png")  # 지금 필수
#np.save(os.path.join(save_dir,"segmentation_mask.npy"), seg)

vis = vis_parsing_maps(img, seg)
cv2.imwrite("vis_check.png", vis)
