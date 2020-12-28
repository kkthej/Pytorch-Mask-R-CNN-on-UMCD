"""Run a demo

Usage:
1. Single image: python demo.py image  --model PATH --input IMAGE_PATH
2. Video or webcam: python demo.py video --model PATH --input PATH_OR_CAM_ID
"""
import os
import colorsys
import random
import time
import cv2
import argparse
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from skimage.measure import find_contours
from PIL import Image

from core.model_factory import get_instance_segmentation_model
from core.tracking.sort import Sort


CLASS_MAP = {
    1: 'tyre',
    2: 'car',
    3: 'man',
    4: 'box',
    5: 'big_box',
    6: 'small_box',
    7: 'suitcase',
    8: 'bag',
    9: 'metal_suitcase',
    10: 'cylinder'
}


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def draw_instances(image, boxes, masks, classes, scores, colors):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [num_instances, height, width]
    class_ids: [num_instances]
    labels: list of class names of the dataset
    scores: (optional) confidence scores for each box
    colors: TODO
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[0]

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.copy()
    for i in range(N):
        score = scores[i] if scores is not None else None
        color = colors[i]
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(masked_image, (x1, y1), (x2, y2),
                      [int(x*255) for x in (color)], 2)

        # Label
        class_id = classes[i]
        label = CLASS_MAP[class_id]
        text = "{}: {}%".format(label, int(score*100))

        yyy = y1 - 16
        if yyy < 0:
            yyy = 0

        cv2.putText(masked_image, text, (x1, yyy), cv2.FONT_HERSHEY_SIMPLEX,
                    1., [int(x*255) for x in (color)], 1, cv2.LINE_AA)

        # Mask
        mask = masks[i, :, :]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            pts = np.array(verts.tolist(), np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(masked_image, [pts], True,
                          [int(x*255) for x in (color)], 2)
    return masked_image.astype(np.uint8)


def detect(model, image, device, conf_threshold=0.5, mask_threshold=0.5):
    """Detect on single image"""
    image = F.to_tensor(image)
    model.eval()
    with torch.no_grad():
        prediction = model([image.to(device)])

    scores = prediction[0]['scores'].cpu().numpy()
    conf_mask = scores > conf_threshold

    masks = (prediction[0]['masks'] > mask_threshold).byte().cpu().numpy()
    masks = np.array([m[0] for m in masks])
    boxes = prediction[0]['boxes'].to(torch.int64).cpu().numpy()
    classes = prediction[0]['labels'].byte().cpu().numpy()

    boxes = boxes[conf_mask]
    masks = masks[conf_mask]
    classes = classes[conf_mask]
    scores = scores[conf_mask]
    return boxes, masks, classes, scores


def detect_image(model, image, device,
                 conf_threshold=0.5, mask_threshold=0.5, out_file=None):
    """Detect and display result on a single image."""
    start = time.time()
    boxes, masks, classes, scores = detect(model, image, device,
                                           conf_threshold, mask_threshold)
    print('Detection time: {:.3f}s'.format(time.time() - start))
    image = np.array(image)
    cmap = random_colors(len(CLASS_MAP) + 1)
    colors = [cmap[class_id] for class_id in classes]
    masked_image = draw_instances(image, boxes, masks, classes, scores, colors)

    cv2.imshow('img', masked_image)
    cv2.waitKey(0)

    if out_file:
        cv2.imwrite(out_file, masked_image)
    print(f'Saved the masked image as {out_file}')


def detect_video(model, video, device, conf_threshold=0.5,
                 mask_threshold=0.5, out_file=None):
    """Detect and display result on a video or webcam"""
    cmap = random_colors(len(CLASS_MAP) + 1)
    cap = cv2.VideoCapture(video)
    writer = None
    while True:
        ret, image_raw = cap.read()
        if not ret:
            break

        image = Image.fromarray(image_raw).convert('RGB') 
        start = time.time()
        boxes, masks, classes, scores = detect(model, image, device,
                                               conf_threshold,
                                               mask_threshold)
        print('FPS: {:3d}'.format(int(1. / (time.time() - start))))
        colors = [cmap[class_id] for class_id in classes]
        masked_image = draw_instances(image_raw, boxes, masks,
                                      classes, scores, colors)
        cv2.imshow('img', masked_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if out_file and writer is None:
            h, w = image_raw.shape[:2]
            writer = cv2.VideoWriter(out_file,
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                     24, (w, h))
        if writer:
            writer.write(masked_image)
    cap.release()
    if writer:
        writer.release()
        print(f'Saved masked video as {out_file}')
    cv2.destroyAllWindows()


def detect_video_with_tracking(model, video, device, conf_threshold=0.5,
                               mask_threshold=0.5, out_file=None):
    """Detect, track direction and display result on a video or webcam."""
    cmap = random_colors(len(CLASS_MAP) + 1)
    cap = cv2.VideoCapture(video)
    writer = None

    tracker = Sort()
    while True:
        ret, image_raw = cap.read()
        if not ret:
            break

        image = Image.fromarray(image_raw).convert('RGB') 
        start = time.time()
        boxes, masks, classes, scores = detect(model, image,
                                               device, mask_threshold)
        print('FPS: {:3d}'.format(int(1. / (time.time() - start))))

        # Update multi-tracker
        colors = [cmap[class_id] for class_id in classes]
        
        if boxes.shape[0] > 0:
            track_bb_ids, traces = tracker.update(boxes)
            for i, box_and_id in enumerate(track_bb_ids):
                box = list(map(int, box_and_id[:4]))
                track_id = int(box_and_id[-1])
                trace = traces[i]
                cv2.putText(image_raw, str(track_id), (box[2] + 10, box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.,
                            colors[i], 1, cv2.LINE_AA)

                # calculate changing direction and draw history
                if len(trace) >= 5:
                    top_trace = [b[1] for b in list(trace)[-5:]]
                    direction_hist = (np.diff(top_trace) > 0).sum()
                    direction = 'forward' if direction_hist < 3 else 'backward'
                    cv2.putText(image_raw, direction, (box[2] + 10, box[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                1, cv2.LINE_AA)
                    for b in trace:
                        b = list(map(int, b))
                        center = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
                        cv2.circle(image_raw, center, 2, (0, 0, 255), -1)

        masked_image = draw_instances(image_raw, boxes, masks,
                                      classes, scores, colors)
        cv2.imshow('img', masked_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if out_file and writer is None:
            h, w = image_raw.shape[:2]
            writer = cv2.VideoWriter(out_file,
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                     24, (w, h))
        if writer:
            writer.write(masked_image)
    cap.release()
    if writer:
        writer.release()
        print(f'Saved masked video as {out_file}')
    cv2.destroyAllWindows()


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--model', default=None, type=str,
                               help='Path to model')
    parent_parser.add_argument('--vis_threshold', default=0.5, type=float,
                               help='Visualization threshold')
    parent_parser.add_argument('--mask_threshold', default=0.5, type=float,
                               help='Mask threshold')
    parent_parser.add_argument('--conf_threshold', default=0.5, type=float,
                               help='Confidence threshold')
    parent_parser.add_argument('--out_dir',
                               default='saved_models/final_output', type=str,
                               help='Result output dir if specified')

    parser = argparse.ArgumentParser(add_help=False)
    subparser = parser.add_subparsers()
    video_parser = subparser.add_parser('video', parents=[parent_parser])
    video_parser.add_argument('--input', default='0', type=str,
                              help='Path to video input or webcam id')
    video_parser.add_argument('--with_tracking',
                              action='store_true',
                              help='Whether to tracking')
    video_parser.set_defaults(input_type='video')

    image_parser = subparser.add_parser('image', parents=[parent_parser])
    image_parser.add_argument('--input', default='', type=str,
                              help='Path to image input')
    image_parser.set_defaults(input_type='image')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Not found model file {args.model}')

    # load model
    num_classes = 11
    device = (torch.device('cuda')
              if torch.cuda.is_available() else torch.device('cpu'))
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    print(f'Loading model from {args.model}')
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model'])

    input_type = args.input_type
    if input_type == 'video':
        video_out_file = None
        if args.out_dir:
            video_out_dir = os.path.join(args.out_dir, 'video_outputs')
            os.makedirs(video_out_dir, exist_ok=True)
            video_out_file = os.path.join(video_out_dir, 'output.avi')
        video = args.input
        if video.isnumeric():
            video = int(video)
        if isinstance(video, str) and not os.path.exists(video):
            raise FileNotFoundError(f'Not found video file {video}')

        if args.with_tracking:
            detect_video_with_tracking(model, video, device,
                                       args.conf_threshold,
                                       args.mask_threshold,
                                       video_out_file)
        else:
            detect_video(model, video, device,
                         args.conf_threshold,
                         args.mask_threshold, video_out_file)
    elif input_type == 'image':
        if not os.path.exists(args.input):
            raise FileNotFoundError(f'Not found image file {args.image}')
        image = Image.open(args.input)
        image_out_file = None
        if args.out_dir:
            image_out_dir = os.path.join(args.out_dir, 'image_outputs')
            os.makedirs(image_out_dir, exist_ok=True)
            image_out_file = os.path.join(image_out_dir, 'output.jpg')
        detect_image(model, image, device,
                     args.conf_threshold, args.mask_threshold,
                     image_out_file)
    else:
        raise ValueError('Unsupported input type')


if __name__ == '__main__':
    main()
