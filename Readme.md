## SOLUTION (DBAI DL Challenge)
- Model Used: Tiny-YOLOv3
- Framework: Darknet
- Hyper parameter and setting used for training:
  -  Learning Rate: 0.001
  -  Momentum: 0.9
  -  Decay: 0.0005
  -  Used transfer learning to used previously trained weights of darknet53.conv.74 
- Training Logs: train_log.txt
- Model weights after 550: [backup/yolov3-tiny_550.weights](https://drive.google.com/file/d/1pjI6QFcXpxjatNyCQIL5rNr1MJ8fcseF/view?usp=sharing) 

### Directory architecture
```
- darknet
     |- aitask
          |- aitask.data
          |- classes.names
          |- train.txt (*.jpg files path for training)
          |- test.txt  (*.jpg files path for testing)
     |- dataset
     |     |-a.jpg (for sample file a.jpg)
     |     |-a.txt (file containing class and co-ordinate details in YOLO-style)
     |- darknet53.conv.74
     |-test_data
     |-cfg/yolov3-tiny.cfg (use the file provided in this repository, it is customized for 10 classes.)
     |-backup/ (this directory will contain the saved model-weights after few epochs)
     |-examples/detector.c (change the number of epochs after which the weights will get saved.)
     
```     
### Training:
```
device:~/Desktop/darknet$ ./darknet detector train aitask/aitask.data cfg/yolov3-tiny.cfg darknet53.conv.74
```

### Inference
```
device:~/Desktop/darknet$ ./darknet detector test aitask/aitask.data cfg/yolov3-tiny.cfg backup/yolov3-tiny_550.weights test_data/video_rms_2020-10-31_16-*.jpg -thresh 0.04
```

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
