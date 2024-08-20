import os
import cv2
import argparse

import matplotlib.pyplot as plt


def plot_image_with_bboxes(image, file_path, is_pred=False, threshold=0.0):

    len_boxes = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split()
        if is_pred:
            label, x1, y1, x2, y2, score = parts
            score = float(score)
            if score < threshold:
                continue  # Skip bboxes below the threshold
            color = (0, 255, 0)  # Green for predicted bboxes
            text = f'{score:.2f}'
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
            len_boxes += 1
        else:
            if 'bbGt' in line: 
                continue
            label, x, y, w, h, *rest = parts
            color = (0, 0, 255)  # Red for ground truth bboxes
            text = label
            x1 = float(x)
            y1 = float(y)
            x2 = x1 + float(w)
            y2 = y1 + float(h)
            len_boxes += 1
        
        if int(y1) < 10:
            text_y = int(y2)
        else:
            text_y = int(y1) - 10
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, len_boxes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.0)
    args = parser.parse_args()

    #########  set parameters  #########
    os.makedirs(f'/SSDe/heeseon/src/C2Former/output/visualize/C2Former/{args.threshold}', exist_ok=True)
    os.makedirs(f'/SSDe/heeseon/src/C2Former/output/visualize/C2Former/{args.threshold}/diff', exist_ok=True)
    
    root_path_pred = '/SSDe/heeseon/src/C2Former/KaistResults/C2Former/det'
    root_path_gt = '/ailab_mat/dataset/KAIST_PED/kaist-cvpr15/annotations'
    root_path_img = '/ailab_mat/dataset/KAIST_PED/kaist-cvpr15/images'


    file_list_pred = os.listdir(root_path_pred)
    file_list_pred.sort()

    for idx, pred in enumerate(file_list_pred):
        print(pred)
        _info = pred.split('.')[0]
        _set, _ver, _img = _info.split('_')

        visible_pred_path = f'{root_path_pred}/{pred}'

        visible_gt_path = f'{root_path_gt}/{_set}/{_ver}/visible/{_img}.txt'
        thermal_gt_path = f'{root_path_gt}/{_set}/{_ver}/lwir/{_img}.txt'

        visible_img_path = f'{root_path_img}/{_set}/{_ver}/visible/{_img}.jpg'
        thermal_img_path = f'{root_path_img}/{_set}/{_ver}/lwir/{_img}.jpg'

        #########  read image and bbox  #########
        visible_img = cv2.imread(visible_img_path)
        thermal_img = cv2.imread(thermal_img_path)

        #########  visualize  #########
        visible_gt, len_gt_boxes = plot_image_with_bboxes(visible_img.copy(), visible_gt_path, threshold=args.threshold)  # RGB 이미지 + GT BBox
        thermal_gt, len_gt_boxes = plot_image_with_bboxes(thermal_img.copy(), thermal_gt_path, threshold=args.threshold)  # Thermal 이미지 + GT BBox
        visible_pred, len_pred_boxes = plot_image_with_bboxes(visible_img.copy(), visible_pred_path, is_pred=True, threshold=args.threshold)  # RGB 이미지 + Pred BBox

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(thermal_gt, cv2.COLOR_BGR2RGB))
        plt.title('Thermal Image + GT BBox')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(visible_gt, cv2.COLOR_BGR2RGB))
        plt.title('RGB Image + GT BBox')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(visible_pred, cv2.COLOR_BGR2RGB))
        plt.title('RGB Image + Pred BBox')

        plt.suptitle(f'{_info}')
        plt.tight_layout()
        plt.savefig(f'/SSDe/heeseon/src/C2Former/output/visualize/C2Former/{args.threshold}/{_info}.png')

        if len_gt_boxes != len_pred_boxes:
            plt.savefig(f'/SSDe/heeseon/src/C2Former/output/visualize/C2Former/{args.threshold}/diff/{_info}.png')

        plt.close()









