"""Visualizes CNN activation maps to see where the CNN focuses on to extract features.
Reference:
    - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
      performance of convolutional neural networks via attention transfer. ICLR, 2017
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
"""

import sys
import os
import numpy as np
import math
import os.path as osp
import cv2
import torch
from torch.nn import functional as F

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), ".."))

import data
from model import make_model
from optim import make_optimizer, make_scheduler
from option import args
import utils.utility as utility
from utils.model_complexity import compute_model_complexity
import yaml


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(model, test_loader, save_dir, width, height, use_gpu, img_mean=None, img_std=None):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    for target in list(test_loader.keys()):
        data_loader = test_loader[target]["query"]  # only process query images
        # original images and activation maps are saved individually
        actmap_dir = osp.join(save_dir, "actmap_" + target)

        if not osp.exists(actmap_dir):
            os.makedirs(actmap_dir)

        print("Visualizing activation maps for {} ...".format(target))

        for batch_idx, data in enumerate(data_loader):
            imgs, paths = data[0], data[-1]
            if use_gpu:
                imgs = imgs.cuda()

            # forward to get convolutional feature maps
            try:
                outputs_list = model(imgs)
            except TypeError:
                raise TypeError(
                    'forward() got unexpected keyword argument "return_featuremaps". '
                    'Please add return_featuremaps as an input argument to forward(). When '
                    'return_featuremaps=True, return feature maps only.'
                )

            for j, output in enumerate(zip(*outputs_list)):
                n_fmaps = len(output)
                grid_img = 255 * np.ones((height, (n_fmaps + 1) * width + n_fmaps * GRID_SPACING, 3), dtype=np.uint8)

                # RGB image
                if use_gpu:
                    imgs = imgs.cpu()
                if j>= imgs.size()[0]:
                    break
                img = imgs[j, ...]
                for t, m, s in zip(img, img_mean, img_std):
                    t.mul_(s).add_(m).clamp_(0, 1)
                img_np = np.uint8(np.floor(img.numpy() * 255))
                # (c, h, w) -> (h, w, c)
                img_np = img_np.transpose((1, 2, 0))

                path = paths[j]
                imname = osp.basename(osp.splitext(path)[0])
                for output_number, outputs in enumerate(output):
                    outputs = (outputs**2).sum(0)
                    h, w = outputs.size()
                    outputs = outputs.view(h * w)
                    outputs = F.normalize(outputs, p=2, dim=0)
                    outputs = outputs.view(h, w)
                    
                    # Original:
                    # if outputs.size()[0]!=24:
                    #     z=torch.zeros(12,8).cuda()
                    #     if output_number == 4:
                    #         outputs=torch.cat((outputs,z),0) 
                    #     if output_number == 5 :
                    #         outputs=torch.cat((z,outputs),0)
                    
                    # BayWald: 20, Wildpark: 15
                    if outputs.size()[0]!=11:
                        if output_number == 4:
                            outputs=torch.cat((outputs,torch.zeros(math.ceil(outputs.size()[0]), outputs.size()[1]).cuda()), 0) 
                        if output_number == 5 :
                            outputs=torch.cat((torch.zeros(math.floor(outputs.size()[0]), outputs.size()[1]).cuda(), outputs), 0)

                    if use_gpu:
                        outputs = outputs.cpu()

                    # activation map
                    am = outputs.numpy()
                    try:
                        am = cv2.resize(am,(width,height))
                    except:
                        print(path)
                        break

                    am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
                    am = np.uint8(np.floor(am))
                    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                    # overlapped
                    overlapped = img_np * 0.45 + am * 0.55
                    overlapped[overlapped > 255] = 255
                    overlapped = overlapped.astype(np.uint8)

                    # save images in a single figure (add white spacing between images)
                    # from left to right: original image, activation map, overlapped image

                    grid_img[:, :width, :] = img_np[:, :, ::-1]
                    grid_img[:,(output_number + 1) * width + (output_number + 1) * GRID_SPACING:(output_number + 2) * width + (output_number + 1) * GRID_SPACING, :] = overlapped
                    # grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
                cv2.imwrite(osp.join(actmap_dir, imname  + '.jpg'), grid_img)

            if (batch_idx + 1) % 10 == 0:
                print('- done batch {}/{}'.format(batch_idx + 1, len(data_loader)))


def main():
    if args.config != "":
        with open(args.config, "r") as f:
            config = yaml.load(f)
        for op in config:
            setattr(args, op, config[op])

    ckpt = utility.checkpoint(args)
    loader = data.VideoDataManager(args) if args.video else data.ImageDataManager(args)
    model = make_model(args, ckpt)
    optimzer = make_optimizer(args, model)

    start = -1
    if args.load != "":
        start, model, optimizer = ckpt.resume_from_checkpoint(osp.join(ckpt.dir, "model-best.pth"), model, optimzer)
        start = start - 1
    if args.pre_train != "":
        ckpt.load_pretrained_weights(model, args.pre_train)

    scheduler = make_scheduler(args, optimzer, start)

    ckpt.write_log("[INFO] Model parameters: {com[0]} flops: {com[1]}".format(com=compute_model_complexity(model, (1, 3, args.height, args.width))))

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    save_dir = ckpt.dir

    visactmap(model, loader.testloader, save_dir, args.width, args.height, use_gpu)


if __name__ == "__main__":
    main()
