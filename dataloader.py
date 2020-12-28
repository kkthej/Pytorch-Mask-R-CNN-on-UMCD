"""Abstract class for UMCD dataset loader"""
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO


class UMCDDataset(torch.utils.data.Dataset):
    """Abstract for UMCD dataset loader.

    Args
    :root: The root directory.
    :transforms: The transform object.
    """
    def __init__(self, image_dir, anno_path, transforms):
        self.image_dir = image_dir
        self.anno_path = anno_path
        self.transforms = transforms
        self.coco = COCO(anno_path)
        self.image_ids = self.coco.getImgIds()
        self.image_info = self.coco.loadImgs(self.image_ids)
        self.cate_ids = self.coco.getCatIds()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        anno_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cate_ids)
        anns = self.coco.loadAnns(anno_ids)

        # find mask file path coressponding to the image
        image_info = self.coco.loadImgs(image_id)[0]
        image_filename = image_info['file_name']
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        boxes = []
        labels = []
        area = []
        iscrowd = []
        labels = []
        instance_masks = []
        for ann in anns:
            box_data = ann['bbox']
            box_data = [box_data[0], box_data[1],
                        box_data[0] + box_data[2],
                        box_data[1] + box_data[3]]
            boxes.append(box_data)
            class_id = ann['category_id']
            labels.append(class_id)
            area.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

            mask = Image.new('1',
                             (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in ann['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
            bool_array = np.array(mask) > 0
            instance_masks.append(bool_array)

        masks = np.array(instance_masks).astype('uint8')
        labels = np.array(labels, dtype=np.int32)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([image_id])
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(image, target)

        return img, target

    def __len__(self):
        return len(self.image_ids)
