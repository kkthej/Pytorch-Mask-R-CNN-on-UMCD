"""Train Pytorch Faster-RCNN on UMCD dataset.
This script is highly inspired by 
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Usage: python train.py
"""
import os
import glob
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt

from core import utils
from core.transforms import get_transform
from core.engine import train_one_epoch, evaluate
from core.model_factory import get_instance_segmentation_model
from dataloader import UMCDDataset


def write_result(data, result_path):
    """
    """
    with open(result_path, 'wt') as f:
        for row in data:
            f.write(','.join(map(str, row)) + '\n')


def plot_result(result_path, output_path=None):
    """
    """
    data = []
    with open(result_path, 'rt') as f:
        for line in f:
            line = line.strip().split(',')
            data.append([float(x) for x in line])
    
    data = np.array(data, dtype=np.float32)
    metrics = ['loss', 'mAP@0.5', 'mAP@.5:.95', 'seg mAP@0.5', 'seg mAP@.5:.95']
    indexes = [0, 2, 1, 14, 13]
    
    cols = 3
    rows = max(2, len(metrics) // cols + int(len(metrics) % cols > 0))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    for idx, (value_idx, metric) in enumerate(zip(indexes, metrics)):
        r = idx // cols
        c = idx % cols
        axes[r, c].plot(np.arange(data.shape[0]), data[:, value_idx])
        axes[r, c].set_title(metric)
    
    if output_path is None:
        output_path = os.path.splitext(result_path)[0] + '.png'
    
    plt.savefig(output_path)


def main():
    parser = argparse.ArgumentParser('Pytorch Mask R-CNN')
    parser.add_argument('--data_dir',
                        default='data/git_repos/cocosynth/datasets/UMCD',
                        help='Dataset root dir')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to checkpoint')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--save_step', default=1, type=int,
                        help='Save the model every')
    parser.add_argument('--eval_step', default=1, type=int,
                        help='Evaluate the model every')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of parallels data loaders')
    parser.add_argument('--save_model', default='saved_models', type=str,
                        help='Save the model to this path')
    parser.add_argument('--log_dir', default='runs',
                        help='Log folder')
    args = parser.parse_args()

    # use our dataset and defined transformations
    data_dir = args.data_dir
    train_image_dir = os.path.join(data_dir, 'train', 'images')
    train_anno_path = os.path.join(data_dir, 'train', 'coco_instances.json')
    dataset = UMCDDataset(train_image_dir,
                          train_anno_path,
                          get_transform(train=True))

    val_image_dir = os.path.join(data_dir, 'val', 'images')
    val_anno_path = os.path.join(data_dir, 'val', 'coco_instances.json')
    dataset_test = UMCDDataset(val_image_dir,
                               val_anno_path,
                               get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    device = (torch.device('cuda')
              if torch.cuda.is_available() else torch.device('cpu'))

    # our dataset has 11 classes only - background and 10 objects
    num_classes = 11

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
    # load checkpoint if need
    start_epoch = 0
    if args.checkpoint:
        print(f'Restoring model from {args.checkpoint}')
        ckpt = torch.load(args.checkpoint)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    # prepare logging folder
    os.makedirs(args.log_dir, exist_ok=True)
    result_path = os.path.join(args.log_dir, 'result.txt')

    # train the model
    results = []
    for epoch in range(start_epoch+1, args.epochs+1):
        epoch_loss = train_one_epoch(model, optimizer, data_loader,
                                     device, epoch, print_freq=10)
        lr_scheduler.step()

        # evaluation
        evaluator = evaluate(model, data_loader_test, device=device)
        
        # Logging
        bbox_stats = evaluator.coco_eval['bbox'].stats
        seg_stats = evaluator.coco_eval['segm'].stats
        
        results.append([epoch_loss, *bbox_stats, *seg_stats])

        if epoch % args.save_step == 0:
            if not os.path.exists(args.save_model):
                os.mkdir(args.save_model)

            for path in glob.glob(os.path.join(args.save_model, '*')):
                os.remove(path)
            save_path = os.path.join(args.save_model,
                                     'model_{:06d}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)
            print(f'Saved as {save_path}')
    
    write_result(results, result_path)
    plot_result(result_path)

    # save final model
    save_path = os.path.join(args.save_model, 'model_final.pth')
    torch.save(model.state_dict(), save_path)
    print('Done!')


if __name__ == '__main__':
    main()
