from PIL import Image
from argparse import ArgumentParser
import glob
import os
from tqdm import tqdm
import sys
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps


parser = ArgumentParser()
parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
parser.add_argument('--faceParsing_ckpt', type=str, default="/home/seclab/hdd3/REFace/Other_dependencies/face_parsing/79999_iter.pth")  
		
parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file, '
                                                                    'this option is valid when --faceParsing_ckpt=segenext')
parser.add_argument('--image_root', type=str, default="/home/seclab/hdd1/code/REFace/datasets/UADFV/UADFVMask/UADFV-img-reshaped")  
parser.add_argument('--save_root', type=str, default="/home/seclab/hdd1/code/REFace/datasets/UADFV/UADFVMask/UADFV-mask-reshaped/Overall_mask")
parser.add_argument('--vis_root', type=str, default="/home/seclab/hdd1/code/REFace/datasets/UADFV/UADFVMask/UADFV-mask-reshaped/Overall_mask_vis")
parser.add_argument('--save_vis', action='store_true')
parser.add_argument('--seg12', action='store_true')

args = parser.parse_args()

os.makedirs(args.save_root, exist_ok=True)
if args.save_vis:
    os.makedirs(args.vis_root, exist_ok=True)

faceParsing_model = init_faceParsing_pretrained_model(args.faceParser_name, args.faceParsing_ckpt, args.segnext_config)    

imgs = sorted(glob.glob(os.path.join(args.image_root, "*.png")), reverse=True)
#print(f"Found {len(imgs)} images")
#print(imgs[:5])

for img in tqdm(imgs, total=len(imgs)):
    filename = os.path.basename(img)
    
    pil_im = Image.open(img).convert("RGB").resize((512, 512), Image.BILINEAR)
    mask = faceParsing_demo(faceParsing_model, pil_im, convert_to_seg12=args.seg12, model_name=args.faceParser_name)
    Image.fromarray(mask).save(os.path.join(args.save_root, filename))
        
    if args.save_vis:
        mask_vis = vis_parsing_maps(pil_im, mask)
        Image.fromarray(mask_vis).save(os.path.join(args.vis_root, filename))


