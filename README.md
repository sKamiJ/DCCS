# DCCS
Official PyTorch implementation for ECCV'20 paper: 
[Deep Image Clustering with Category-Style Representation](https://arxiv.org/pdf/2007.10004.pdf)

## Package dependencies

- python >= 3.6
- pytorch == 1.2.0
- torchvision == 0.4.0
- scikit-learn == 0.21.3
- tensorboardX
- matplotlib
- numpy
- scipy

## Create the environment with Anaconda
```shell
$ conda create -n dccs python=3.6
$ source activate dccs
$ conda install pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.0 -c pytorch
$ conda install scikit-learn=0.21.3
$ pip install tensorboardX
$ conda install matplotlib
```

## Prepare datasets

For MNIST, Fashion-MNIST, CIFAR-10 and STL-10, you can download the datasets using torchvision. 

For example, you can download CIFAR-10 with
> torchvision.datasets.CIFAR10('path/to/dataset', download=True)

For ImageNet-10, you can download ImageNet, select the images of 10 classes listed in './data/imagenet10_classes.txt', and resize the images to 96x96 pixels.

## Command to run DCCS

You can run DCCS on MNIST with

```shell
$ CUDA_VISIBLE_DEVICES=0 python train.py --dataset-type=MNIST --dataset-path=path/to/dataset --beta-aug=2 
```

For CIFAR-10, you can use 
```shell
$ CUDA_VISIBLE_DEVICES=0 python train.py --dataset-type=CIFAR10 --dataset-path=path/to/dataset --beta-aug=4 
```

## Citation

If you are interested in our paper, please cite:
```
@inproceedings{zhao2020deep,
  title={Deep Image Clustering with Category-Style Representation},
  author={Zhao, Junjie and Lu, Donghuan and Ma, Kai and Zhang, Yu and Zheng, Yefeng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```