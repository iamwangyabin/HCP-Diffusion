import argparse
import json
import os

from hcpdiff.utils.img_size_tool import types_support

parser = argparse.ArgumentParser(description='Generate Image Captions for Stable Diffusion Training')
parser.add_argument('--data_root', type=str, required=True, help='Root directory of the image folders')
parser.add_argument('--with_imgs', action="store_true", help='Include captions for image files')
args = parser.parse_args()

def get_txt_caption(path):
    with open(path, encoding='utf8') as f:
        return f.read().strip()

for subdir in next(os.walk(args.data_root))[1]:
    subdir_path = os.path.join(args.data_root, subdir)
    captions = {}

    for file in os.listdir(subdir_path):
        file_name, file_ext = os.path.splitext(file)
        file_ext = file_ext.lstrip('.')  # Remove the dot from the extension for consistency

        if args.with_imgs:
            if file_ext in types_support:
                txt_path = os.path.join(subdir_path, f'{file_name}.txt')
                if os.path.exists(txt_path):
                    captions[file] = get_txt_caption(txt_path)
        else:
            if file_ext == 'txt':
                captions[file] = get_txt_caption(os.path.join(subdir_path, file))

    if captions:
        with open(os.path.join(subdir_path, 'image_captions.json'), "w", encoding='utf8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)

        print(subdir_path)


