# StrAttack

The code is for paper: 'Structured Adversarial Attack: Towards General Implementation and Better Interpretability' which accepted in ICLR 2019
(https://openreview.net/forum?id=BkgzniCqY7) by Kaidi Xu*, Sijia Liu*, Pu Zhao, Pin-Yu Chen, Huan Zhang, Quanfu Fan, Deniz Erdogmus, Yanzhi Wang, Xue Lin (* means equal contribution)

```
@inproceedings{
xu2018structured,
title={Structured Adversarial Attack:  Towards General Implementation and Better Interpretability},
author={Kaidi Xu and Sijia Liu and Pu Zhao and Pin-Yu Chen and Huan Zhang and Quanfu Fan and Deniz Erdogmus and Yanzhi Wang and Xue Lin},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=BkgzniCqY7},
}
```


Experiment Setup
-------------------------------------
To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)


and put the `imgs` folder in `../imagesnetdata`. This path can be changed
in `setup_inception.py`.

To download the inception model:

```
python3 setup_inception.py
```


To train CIFAR10 and MNIST model:
run 
```
python3 trainmodel.py -d all
```

StrAttack:

run 
```
python3 test_attack_iclr.py
```
You can change methods, dataset or any hyperparameter in args or add in command line.


