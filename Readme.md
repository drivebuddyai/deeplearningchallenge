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
