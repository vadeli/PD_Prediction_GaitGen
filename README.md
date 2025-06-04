# PD_Prediction_GaitGen

Unzip everything in this link (https://drive.google.com/drive/folders/11z8bNvfcLdyZucHwIZNS-6o9gYFGkBzG?usp=sharing) to root of the project. 

The root directory structure should look:
```
root/
└── checkpoints/
        ├── ...
    └── data/
        ├── gaitgen/
            ├── data_gaitgen.pkl
            ├── ...
        ├── dpgam
            ├── HumanML3Drep
                ├── ...
            ├── ...
    ├── exp/ 
        ├── ...
    ├── models/ 
        ├── ...
    ├── utils/ 
        ├── ...
```

To run prediction use these commands:
```
 python downstream_task_train_test.py --model checkpoints/classifier_pdgam_synthetic.pth --exp_dataname pdgam_synthetic
 python downstream_task_train_test.py --model checkpoints/classifier_pdgam.pth --exp_dataname pdgam
```

To train:
```
 python downstream_task_train_test.py --exp_dataname pdgam_synthetic --numepoch 4000 --lr 0.00001
 python downstream_task_train_test.py --exp_dataname pdgam --numepoch 1500 --lr 0.0001
```