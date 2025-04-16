import os
import shutil

src_root = '/home/seclab/hdd1/code/DeepfakeBench/datasets/UADFV/real/frames'
dst_root = '/home/seclab/hdd1/code/REFace/datasets/UADFV/UADFVMask/UADFV_img'
os.makedirs(dst_root, exist_ok=True)

for video_folder in os.listdir(src_root):
    video_path = os.path.join(src_root, video_folder)
    
    if not os.path.isdir(video_path):
        continue
    
    for frame_name in os.listdir(video_path):
        src_frame_path = os.path.join(video_path, frame_name)
        
        new_frame_name = f"{video_folder}_{frame_name}"
        dst_frame_path = os.path.join(dst_root, new_frame_name)
        
        shutil.copy2(src_frame_path, dst_frame_path)

