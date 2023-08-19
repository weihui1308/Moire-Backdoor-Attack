import argparse
import os
import platform
import sys
from pathlib import Path

import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, img2label_paths
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
from utils.segment.dataloaders import polygons2masks_overlap, polygons2masks
import numpy as np
from PIL import Image
import torchvision.transforms as T
from moire_plus import parallel_line_moire, rotated_line_moire, curved_line_moire, Moire
from random import choice


def load_masks_with_size(paths, img):
    segments = []
    masks_size = 0.3
    with open(paths, 'r') as label_file:
        lines = label_file.readlines()
        for line in lines:
            if line[0]=='0':
                line = line.strip().split()[1:]#  [0]==class
                line = np.array(line).astype(float)
                line[1::2] *= img.shape[0]
                line[::2] *= img.shape[1]
                segments.append(line)

        masks = polygons2masks(img.shape[0:2], segments, color=1, downsample_ratio=1)
        masks_tmp = np.ones_like(masks) * -1
        for i in range(len(masks)):
            person_height = max(segments[i][1::2]) - min(segments[i][1::2])
            low = int(max(segments[i][1::2]) - (1-((1-masks_size) / 2)) * person_height)#
            top = int(max(segments[i][1::2]) -  ((1-masks_size) / 2) * person_height)
            masks_tmp[i][low:top, :] = 0
    masks = masks + masks_tmp
    masks[masks<0] = 0
    return masks
    
@smart_inference_mode()
def run(
    source=ROOT / 'data/images',
    imgsz=(640, 640),  # inference size (height, width)
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    nosave=False,  # do not save images/videos
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    moire = Moire()
    i = 0
    for path, im, im0s, vid_cap, s in dataset:
        i+=1
        print(i)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]

        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        
        label_path = img2label_paths([path])[0]
        bs, c, h, w = im.size()
        masks = load_masks_with_size(label_path, im0)

        moire_type = choice(['parallel_line_moire', 'rotated_line_moire', 'curved_line_moire'])
        moire_img = moire(masks, im0, moire_type)
    
        im0 = np.clip(moire_img * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path, im0)

        


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/images/train2017/", help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
