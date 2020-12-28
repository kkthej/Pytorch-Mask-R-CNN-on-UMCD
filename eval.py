"""Evaluation script"""
import os
import glob
import argparse
import torch
import torchvision

from core import utils
from core.transforms import get_transform
from core.model_factory import get_instance_segmentation_model
from core.engine import evaluate
from dataloader import UMCDDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str,
                        help='Path to trained model')
    parser.add_argument('--data_dir',
                        default='data/git_repos/cocosynth/datasets/UMCD',
                        help='Dataset root dir')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of parallels data loaders')
    args = parser.parse_args()

    # load data
    data_dir = args.data_dir
    val_image_dir = os.path.join(data_dir, 'val', 'images')
    val_anno_path = os.path.join(data_dir, 'val', 'coco_instances.json')

    dataset_test = UMCDDataset(val_image_dir,
                               val_anno_path,
                               get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    # load model
    device = (torch.device('cuda')
              if torch.cuda.is_available() else torch.device('cpu'))
    num_classes = 11
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    ckpt = torch.load(args.model, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    evaluate(model, data_loader_test, device=device)


if __name__ == '__main__':
    main()
