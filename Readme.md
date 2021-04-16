## SOLUTION: DBAI DL Challenge
- Model Used: Tiny-YOLOv3 (Selected because of the available system configuration)
- Framework: Darknet
- Hyper parameter and setting used for training:
  -  Training on: CPU (As available system does not contain GPUs for training)
  -  Learning Rate: 0.001
  -  Momentum: 0.9
  -  Decay: 0.0005
  -  Batch: 24 (set in file cfg/yolov3-tiny.cfg, LINE 3)
  -  Subdivisions: 8 (set in file cfg/yolov3-tiny.cfg, LINE 4)
  -  Used transfer learning with trained weights darknet53.conv.74 (wget https://pjreddie.com/media/files/darknet53.conv.74)
- Training Logs: train_log.txt
- Model weights after 550: [backup/yolov3-tiny_550.weights](https://drive.google.com/file/d/1pjI6QFcXpxjatNyCQIL5rNr1MJ8fcseF/view?usp=sharing) 

#### System Configuration For Training:
- Operating System: Ubuntu 20.04.2 LTS
- RAM: 7.6 GiB
- Processor: Intel® Core™ i5-7200U CPU @ 2.50GHz × 4 
- Graphics: AMD® Hainan / Mesa Intel® HD Graphics 620 (KBL GT2)

### Directory architecture
```
- darknet
     |- aitask
     |     |- aitask.data
     |     |- classes.names
     |     |- train.txt (*.jpg files path for training)
     |     |- test.txt  (*.jpg files path for testing)
     |     
     |- dataset
     |     |-a.jpg (for sample file a.jpg)
     |     |-a.txt (file containing class and co-ordinate details in YOLO-style)
     |     
     |- darknet53.conv.74
     |-test_data (directory containing images for testing)
     |-cfg/yolov3-tiny.cfg (use the file provided in this repository, it is customized for 10 classes.)
     |-backup/ (this directory will contain the saved model-weights after few epochs)
     |-examples/detector.c (change the number of epochs after which the weights will get saved.)
     
```     

### Model Architecture:
```
layer     filters    size              input                output
    0 conv     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16  0.150 BFLOPs
    1 max          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
    2 conv     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32  0.399 BFLOPs
    3 max          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
    4 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64  0.399 BFLOPs
    5 max          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
    6 conv    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128  0.399 BFLOPs
    7 max          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
    8 conv    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256  0.399 BFLOPs
    9 max          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
   10 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
   11 max          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
   12 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   13 conv    256  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 256  0.089 BFLOPs
   14 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
   15 conv     45  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x  45  0.008 BFLOPs
   16 yolo
   17 route  13
   18 conv    128  1 x 1 / 1    13 x  13 x 256   ->    13 x  13 x 128  0.011 BFLOPs
   19 upsample            2x    13 x  13 x 128   ->    26 x  26 x 128
   20 route  19 8
   21 conv    256  3 x 3 / 1    26 x  26 x 384   ->    26 x  26 x 256  1.196 BFLOPs
   22 conv     45  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x  45  0.016 BFLOPs
   23 yolo
```

### Training:
```
sparrow@sparrow-bot:~/Desktop/darknet$ ./darknet detector train aitask/aitask.data cfg/yolov3-tiny.cfg darknet53.conv.74
```

### Inference
```
sparrow@sparrow-bot:~/Desktop/darknet$ ./darknet detector test aitask/aitask.data cfg/yolov3-tiny.cfg backup/yolov3-tiny_550.weights test_data/video_rms_2020-10-31_16-*.jpg -thresh 0.04
```

# Corrections:
- compute_AP.py, this file uses midpoint co-ordinate system to evaluate, while the co-ordinates provided for training in the data/train/gt.pickle are of corners co-ordinates.
- compute_AP.py, earlier the default value for number of classes was wrong (i.e. 91), corrected to 10.
- Some images in the training dataset were of 0Kb size.(removed them)

# Observations:
- Larger number of epochs are required for training.
- Better hardware configuration are needed for faster training.
- Better Deep Learning models can be used provided one have better hardware.
- Some synthetic data can be creating by performing image operations like rotation, scaling, mirroring etc. also correspondingly the co-ordinates.

# Inference Result:
- Higher number of epochs and better computing device is required for getting any relevant result on testing data.

---

# DBAI DL Challenge

#### <u>Welcome to DBAI challenge</u>
The goal of the challenge is to train model that would work well on Indian roads.


### Content :

Repo consist of mAP script for evaluation of the model with training data and test data in [data](data) directory


```
<Tree structure>

dl_challenge
├── Readme.md
├── compute_AP.py                 |  # mAP calculation script
└── data                          |
    ├── test                      |
    │   └── dbai_test_data.zip    |  # test data images
    └── train                     |
        ├── dbai_train_data.zip   |  # coco style training data
        └── gt.pickle             |  # ground truth pickle file

```


- [compute_AP.py](compute_AP.py) :Script to compute AP result for evaluation of model
```
# usage 

GT_PICKLE = "path/to/_GT_/pickle_file"
PRD_PICKLE = ""path/to/_PRD_/pickle_file""

compute_mAP(GT_PICKLE,PRD_PICKLE)

# Output mAP values are saved as csv file at the <prd>.pickle file path 
```


#### <u>Pickle file format</u>:
Bellow is the format for predicted.pickle file which will then be used for calculating mAP score using compute_mAp script.

```
[
    [image_name,
            [ 
                [class_id , class_name , confidence , [x1 , y1 , x2 , y2]
                [class_id , class_name , confidence , [x1 , y1 , x2 , y2]
                ....
            ]
     ]
    ...
]
```

### Deliverable
Send us the following details at [hr@drivebuddyai.co](hr@drivebuddyai.co)
- Predicted pickle file for given test dataset
- Document containing details of
  - Model used
  - Hyper parameter and setting used for training
  - Setting used for testing / inference
  - Training log
  - Approach
  - Your observations and conclusion

### Evaluation

Candidate will be evaluated by
- mAP score per class
- Approach and model selection
