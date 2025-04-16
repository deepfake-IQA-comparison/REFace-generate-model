from PIL import Image
import os

# file_path = '/home/seclab/hdd1/code/DeepfakeBench/datasets/UADFV/real/frames/BiSeNet_mask/000.png'
# img = Image.open(file_path)
# width, height = img.size
# print(f"size: {width}*{height}")

input_dir = '/home/seclab/hdd1/code/REFace/datasets/UADFV/UADFVMask/UADFV-img'
output_dir = '/home/seclab/hdd1/code/REFace/datasets/UADFV/UADFVMask/UADFV-img-resized'
target_size = (512,512)

os.makedirs(output_dir, exist_ok=True)

img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

for filename in os.listdir(input_dir):
    if any(filename.lower().endswith(ext) for ext in img_extensions):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(target_size, resample=Image.LANCZOS)

        save_path = os.path.join(output_dir, filename)
        img_resized.save(save_path)

        print(f"Resized and saved: {filename}")
