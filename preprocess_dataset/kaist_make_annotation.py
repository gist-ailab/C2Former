import os
import cv2
import json
import argparse

import matplotlib.pyplot as plt


def save_annotations_to_json(annotations, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(annotations, json_file, ensure_ascii=False, indent=4)
        print(f"Annotations saved to {filename}")
    except Exception as e:
        print(f"Error saving annotations to JSON: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/ailab_mat/dataset/KAIST_PED/kaist-paired')
    parser.add_argument('--annot_path', type=str, default='/ailab_mat/dataset/KAIST_PED/kaist-paired/kaist-paired/annotations/')
    parser.add_argument('--split', type=str, default='trainval', choices=['trainval', 'test'])
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    image_root_path = f'{args.root_path}/images'
    annot_root_path = f'{args.root_path}/annotations'
    split_path = f'{args.root_path}/splits/{args.split}.txt'

    annotations = {
        "info": {
            "description": "Kaist Paired Dataset",
        },
        "images": [
            # {"id": 242287, "width": 426, "height": 640, "file_name": "xxxxxxxxx.jpg"},
            # {"id": 245915, "width": 640, "height": 480, "file_name": "nnnnnnnnnn.jpg"}
        ],
        "annotations": [
            # {"id": 125686, "category_id": 0, "iscrowd": 0, "segmentation": [[164.81, 417.51,......167.55, 410.64]], 
            # "image_id": 242287, "area": 42061.80340000001, "bbox": [19.23, 383.18, 314.5, 244.46]},
            # {"id": 1409619, "category_id": 0, "iscrowd": 0, "segmentation": [[376.81, 238.8,........382.74, 241.17]], "image_id": 245915, "area": 3556.2197000000015, "bbox": [399, 251, 155, 101]},
            # {"id": 1410165, "category_id": 1, "iscrowd": 0, "segmentation": [[486.34, 239.01,..........495.95, 244.39]], "image_id": 245915, "area": 1775.8932499999994, "bbox": [86, 65, 220, 334]}
        ],
        "categories": [
            {"supercategory": "person","id": 0,"name": "person"}
        ]
    }
    
    image_id, annot_id = 0, 0
    width, height = 512, 640


    with open(split_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            _set, _ver, _img = line.split('/')

            annot_visible_path = f'{annot_root_path}/{_set}/{_ver}/visible/{_img}.txt'
            annot_thermal_path = f'{annot_root_path}/{_set}/{_ver}/lwir/{_img}.txt'

            image_visible_path = f'{image_root_path}/{_set}/{_ver}/visible/{_img}.jpg'
            image_thermal_path = f'{image_root_path}/{_set}/{_ver}/lwir/{_img}.jpg'
            
            if args.debug:
                image_visible = cv2.imread(image_visible_path)
                image_thermal = cv2.imread(image_thermal_path)

            annot_visible, annot_thermal = [], []
            with open(annot_visible_path, 'r') as annot_visible_file:
                annot_visible_lines = annot_visible_file.readlines()
                for i, annot_visible_line in enumerate(annot_visible_lines):
                    if i == 0: continue
                    parts = annot_visible_line.split()
                    _categ = parts[0]
                    x = int(parts[1])
                    y = int(parts[2])
                    w = int(parts[3])
                    h = int(parts[4])

                    annotations['annotations'].append({
                        'id': annot_id,
                        'category_id': 0, 
                        'image_id': image_id,
                        "iscrowd": 0,
                        "area": w * h, 
                        "bbox": [x, y, w, h, 0]     # x, y, w, h, a
                    })

                    if args.debug: cv2.rectangle(image_visible, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    annot_id += 1
            
            annotations['images'].append({
                'id': image_id,
                'width': width,
                'height': height, 
                'file_name': image_visible_path,
                'file_name_tir': image_thermal_path
            })
            
            if args.debug:
                image_visible_rgb = cv2.cvtColor(image_visible, cv2.COLOR_BGR2RGB)

                # 이미지 시각화
                plt.imshow(image_visible_rgb)
                plt.axis('off')  # 축 숨기기

                # 이미지 저장
                plt.savefig('annotated_image.png', bbox_inches='tight', pad_inches=0)
                plt.close()

            image_id += 1

            # if image_id == 16:
            #     break
    
    save_annotations_to_json(annotations, f'{args.split}.json')




# < KAIST homepage >
# -   [2] - C. Li, D. Song, R. Tong, and M. Tang, “Multispectral pedestrian detection via simultaneous detection and segmentation,” in Proc. Brit. Mach. Vision Conf., 2018, pp. 225.1–225.12.
# -   [3] - L. Zhang, X. Zhu, X. Chen, X. Yang, Z. Lei, and Z. Liu, “Weakly aligned cross-modal learning for multispectral pedestrian detection,” in Proc. IEEE Int. Conf. Comput. Vision, 2019, pp. 5126–5136.

# < C2Former >
# - [19] L. Zhang et al., “Weakly aligned feature fusion for multimodal object
# detection,” IEEE Trans. Neural Netw. Learn. Syst., vol. 1, no. 1,
# pp. 1–15, Aug. 2021.
# - [57] J. Liu, S. Zhang, S. Wang, and D. N. Metaxas, “Multispectral deep
# neural networks for pedestrian detection,” 2016, arXiv:1611.02644.
