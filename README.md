# Train CIFAR10,CIFAR100 with Pytorch-lightning
Measure Out-of-Distribution Detection using several methods with [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).
Methods includes [odin](https://arxiv.org/abs/1706.02690)

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
After you have cloned the repository, you can train each dataset of either cifar10, cifar100, SVHN by running the script below.

```bash
python train.py
```

## How to measure Out-of-Distribution Detection
After you train models, run main.py with several arguments.

```bash
python main.py --in_dataset="CIFAR10" --out_dataset="LSUN" --nn="Densetnet_BC"
```

## Detection results

-Baseline

|  In-dist  |  Out-dist  | FPR at TPR 95% | Detection Error |     AUROC     |     AUPR In     |     AUPR Out     |
|:---------:|:----------:|:--------------:|:---------------:|:-------------:|:---------------:|:----------------:|
|   CIFAR   |    LSUN    |      38.4%     |      21.5%      |      94.5%    |      95.7%      |      93.2%       |

-[Odin](https://arxiv.org/abs/1706.02690)

|  In-dist  |  Out-dist  | FPR at TPR 95% | Detection Error |     AUROC     |     AUPR In     |     AUPR Out     |
|:---------:|:----------:|:--------------:|:---------------:|:-------------:|:---------------:|:----------------:|
|   CIFAR   |    LSUN    |      18.1%     |      11.5%      |      97.0%    |      97.4%      |      96.6%       |