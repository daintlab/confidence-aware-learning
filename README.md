# Confidence-Aware Learning for Deep Neural Networks
This repository provides the code for training with *Correctness Ranking Loss* presented in the paper "[Confidence-Aware Learning for Deep Neural Networks](https://arxiv.org/abs/2007.01458)" accepted to ICML2020.

## Getting Started
### Requirements
```
* ubuntu 18.0.4, cuda10
* python 3.6.8
* pytorch >= 1.2.0
* torchvision >= 0.4.0 
```
### Datasets
* CIFAR-10, CIFAR-100, SVHN

## How to Run
### Arguments
| Args 	| Type 	| Description 	| Default|
|---------|--------|----------------------------------------------------|:-----:|
| epochs 	| [int] 	| epochs | 300|
| batch_size 	| [int] 	| batch size| 128|
| data 	| [str] 	| cifar10, cifar100, svhn| cifar10|
| model 	| [str]	| res, dense, vgg| 	res|
| rank_target 	| [str] 	| softmax, entropy, margin| softmax	|
| rank_weight 	| [float] 	| rank_weight| 1.0|
| data_path 	| [str] 	| data path | ./data/  |
| save_path 	| [str] 	| save files path	|  - |
| file_name 	| [str] 	| pretrained file name	|  - |
| gpu 	| [str] 	|  gpu number | 0	|

### Train with Correctness Ranking Loss
```
# Examples 
python main.py --save_path ./res_cifar10/softmax/ --model res --data cifar10 --rank_target softmax --rank_weight 1.0 --gpu 0 
python main.py --save_path ./vgg_cifar100/entropy/ --model vgg --data cifar100 --rank_target entropy --rank_weight 1.0 --gpu 0 
```

### Train baseline models
* Set `rank_weight = 0`.
``` 
# Examples
python main.py --save_path ./res_cifar10/baseline/ --model res --data cifar10 --rank_weight 0.0 --gpu 0 
python main.py --save_path ./vgg_cifar100/baseline/ --model vgg --data cifar100 --rank_weight 0.0 --gpu 0 
```

### Evaluate the trained model
``` 
# Calculate performance measures from the trained model `file_name.pth` located in `save_path`

|---- test.py
|     |---- save_path
|           |---- file_name.pth
|           |---- result.log

python test.py --save_path ./res_cifar10/ --file_name model --model res --data cifar10 --gpu 0 
python test.py --save_path ./vgg_svhn/ --file_name model --model vgg --data svhn --gpu 0
```

## Results
### Performance measures
- Accuracy
- AURC, EAURC
- Expected Calibration Error(ECE)
- Negative Log Likelihood(NLL)
- Brier Score
- AUPR Error, FPR 95% TPR

### Results on CIFAR-100

| Architecture | Dataset | Model | ACC | AURC | AUPR | FPR | ECE | NLL |
|---------|--------|--------|--------|--------|--------|--------|--------|--------------------------------------------------------------------|
| PreActResNet110	| CIFAR100	| Baseline	| 73.32 | 86.54 | 65.37 | 66.42 | 16.39 | 14.93 | 
| PreActResNet110	| CIFAR100	| CRL-softmax	| 74.34 | 72.35 | 68.13 | 61.30 | 11.45 | 10.86 | 
| DenseNet_BC	| CIFAR100	| Baseline	| 75.13 | 72.40 | 66.41 | 62.85 | 12.94 | 11.59 |
| DenseNet_BC	| CIFAR100	| CRL-softmax	| 76.75 | 62.71 | 65.87 | 60.22 | 8.66 | 9.12 |
| VGG16	| CIFAR100	| Baseline	| 73.62 | 77.80 | 68.11 | 62.21 | 19.95 | 18.35 |
| VGG16	| CIFAR100	| CRL-softmax	| 73.84 | 71.98 | 71.04 | 59.06 | 13.92 | 13.03 |

* More results can be found in the paper.

### Citation
```
@inproceedings{moon2020crl,
  title={Confidence-Aware Learning for Deep Neural Networks},
  author={Moon, Jooyoung and Kim, Jihyo and Shin, Younghak and Hwang, Sangheum},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
```

### Contact for issues
- JooYoung Moon, answn3475@ds.seoultech.ac.kr
