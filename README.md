# Train CIFAR10,CIFAR100 with Pytorch-lightning
Measure Out-of-Distribution Detection using several methods with [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning). Methods include [ODIN](https://arxiv.org/abs/1706.02690) etc.

## Requirements
- setup/requirements.txt
```bash
torch 1.5.1
torchvision 0.6.1
pytorch-lightning 0.9.0rc5
tqdm
argparse
pytablewriter
seaborn
enum34
scipy
cffi
sklearn
```

- install requirements using pip
```bash
pip3 install -r setup/requirements.txt
```

## How to train models
After you have cloned the repository, you can train each models with datasets cifar10, cifar100, SVHN. Trainable models are [VGG](https://arxiv.org/abs/1409.1556), [Resnet](https://arxiv.org/abs/1512.03385), [WideResnet](https://arxiv.org/pdf/1605.07146.pdf), [Densenet-BC](https://arxiv.org/pdf/1608.06993.pdf), [Densenet](https://arxiv.org/abs/1608.06993).

```bash
python train.py
```

## How to measure Out-of-Distribution Detection
After you train models, run main.py with several arguments.

```bash
# in-distribution : CIFAR10, out-distribution : LSUN, model : Densenet-BC
python main.py --in_dataset="CIFAR10" --out_dataset="LSUN" --nn="Densetnet_BC"
```

## Detection results

- Densenet-BC

|    Methods    |    In-dist    |    Out-dist   |   FPR at TPR 95%   |   Detection Error  |        AUROC       |       AUPR In      |       AUPR Out      |
|:-------------:|:-------------:|:-------------:|:------------------:|:------------------:|:------------------:|:------------------:|:-------------------:|
|   Baseline    |     CIFAR     |      LSUN     |        38.4%       |        21.5%       |        94.5%       |        95.7%       |        93.2%        |
|[ODIN](https://arxiv.org/abs/1706.02690)|   CIFAR   |    LSUN    |      18.1%     |      11.5%      |      97.0%    |      97.4%      |      96.6%       |