# Pytorch Mask R-CNN
Pytorch Mask-RCNN on custom UMCD dataset.

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
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345|
| Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398|
 |Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.393|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.271|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.289|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.474|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.354|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.376|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.309|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.298|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482|
 
*IoU metric: segm*
|Metric|Area|Max Dets/Value|
|:---|:---|:---|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331|
 |Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.393|
 |Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.377|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.288|
 |Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.343|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.365|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.365|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.285|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.293|
 |Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498|

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