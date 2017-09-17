# refinenet
a tensorflow implement of refinenet. RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation


## Introduction
this is a tensorflow implement of refinenet discribed in [arxiv:1611.06612](https://arxiv.org/abs/1611.06612).I have not finished it yet, this is just a demo, but the model is already able to work.

## prepare
- download the pretrain model of resnet_v1_101.ckpt, you can download it from [here](https://github.com/tensorflow/models/tree/master/slim)
- download the [pascal voc dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- some dependence like cv2, numpy and etc. recommend to install Anaconda

## training
- first, run convert_pascal_voc_to_tfrecords.py to convert training data into .tfrecords, Or you can use the tfrecord I converted In [BaiduYun](http://pan.baidu.com/s/1kVefEIj).Currently, I only use the pascal voc 2012 for training.
- second, run python RefineNet/multi_gpu_train.py, also, you can change some hyper parameters in this file, like the batch size.

## eval
- if you have already got a model, or just download the model I trained on pascal voc.[model](http://pan.baidu.com/s/1kVefEIj).
- put images in demo/ and run python RefineNet/demo.py 

## some result
<img src="/demo/2007_000713.jpg" width=256 height=256 /><img src="/demo/2007_000733.jpg" width=256 height=256 /><img src="/demo/2007_000738.jpg" width=256 height=256 />
<img src="/result/2007_000713.jpg" width=256 height=256 /><img src="/result/2007_000733.jpg" width=256 height=256 /><img src="/result/2007_000738.jpg" width=256 height=256 />
