'''
Doyeon Kim, 2022
'''

import os
from pickletools import optimize
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging as logging
import utils.metrics as metrics
from models.model import SwinMono
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions


def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.save_eval_pngs:
        result_path = os.path.join(args.result_dir, args.exp_name)
        logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)

    print("\n1. Define Model")
    model = SwinMono(max_depth=args.max_depth, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n2. Define Dataloader")
    dataset_kwargs = {'data_path': args.data_path, 'dataset_name': args.dataset,
                          'is_train': False}

    test_dataset = get_dataset(**dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=False)

    print("\n3. Inference & Evaluate")
    for batch_idx, batch in enumerate(test_loader):
        input_RGB = batch['image'].to(device)
        filename = batch['filename']

        with torch.no_grad():
            pred = model(input_RGB)
            torch.cuda.empty_cache()
     
        pred_d = pred['pred_d']

        if args.save_eval_pngs:
            save_path = os.path.join(result_path, filename[0])
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            pred_d = pred_d.squeeze()
            pred_d = pred_d.cpu().numpy() * 256.0
            cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
     
        logging.progress_bar(batch_idx, len(test_loader), 1, 1)
    

    print("Done")


if __name__ == "__main__":
    main()
