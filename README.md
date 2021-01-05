# Pytorch Mask R-CNN
Pytorch Mask-RCNN on custom UMCD dataset.
## First of all
please unzip the file git_repos.rar  and place it in data folder i.e Pytorch-Mask-R-CNN-on-UMCD/data
download zip file from : https://drive.google.com/file/d/1jNGdSdy9kKX2uSYqZpSCcBXcxWq99yb_/view?usp=sharing

## Installation
The project required `torch>=1.6.0` and `torchvision>=0.7.0`.
Other packages could be installed easily by `pip`: `pip install -r requirements.txt`.

Create a directory called `data` and put your `git_repos` directory into. Path to training directory should be `./data/git_repos/cocosynth/datasets/UMCD/train`. Similar to validation set.

Install coco api
```
pip install Cython
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
```
All done!


## Training
To train the model, follow the command:
```
python train.py --batch_size [BATCH_SIZE] --epochs [EPOCHS] --lr [LR]
```
For example,
 ```
 python train.py --batch_size 4 --epochs 50 --lr 1e-3
 ```
Please  run `python train.py --help` for more details.

## for Evaluation and deployment
Download the .pth file from below link and place it in Pytorch-Mask-R-CNN-on-UMCD/saved_models/
https://drive.google.com/file/d/1pfNfskOYlC4AdjZVjpShh4aYeEU2kdnS/view?usp=sharing

# Evaluation
NOTE: Please run evaluation on GPU. CPU is not supported yet.
To evaluate the trained model, follow the below command:
```
python eval.py --model [MODEL_PATH] --data_dir [DATA_DIR]
```
For example, `python eval.py --model saved_models/model_000050.pth`. If nothing wrong, it would
print two tables as the following.
*IoU metric: bbox*
|Metric|Area|Max Dets/Value|
|:---|:---|:---|
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.645|
| Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.698|
 |Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.693|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.671|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.689|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.654|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.676|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.676|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.609|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.698|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682|
 
*IoU metric: segm*
|Metric|Area|Max Dets/Value|
|:---|:---|:---|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.631|
 |Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.693|
 |Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.677|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.733|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.688|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.798|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.643|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.665|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.665|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.685|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.693|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768|

## Deployment
To test on a single image.
```
python demo.py image --model MODEL_PATH --input IMAGE_PATH
```
For example, we'd like to test on image `myimage.jpg`.
```
python demo.py image --model saved_models/model_final.pth --input myimage.jpg
```

To test on video/webcam.
```
python demo.py video --model MODEL_PATH --input VIDEO_OR_CAM_ID [--with_tracking] \
	[--out_dir OUT_DIR]
```
For example, we'd like to test on video `myvideo.mp4`.
```
python demo.py video --model saved_models/model_final.pth --input myvideo.mp4
```
Test with direction tracking.
```
python demo.py video --model saved_models/model_final.pth --input myvideo.mp4 --with_tracking \
	[--out_dir OUT_DIR]
```

Similar to webcam.
```
python demo.py video --model saved_models/model_final.pth --input 0
```
