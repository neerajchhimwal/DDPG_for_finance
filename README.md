# DDPG_for_finance

## Installation steps

```
conda create -n ddpg_torch_gym python=3.7
conda activate ddpg_torch_gym
pip install -r requirements.txt
```

Install pytorch based on system from [here](https://pytorch.org/get-started/locally/):
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Make sure you have an active weights and biases initialization:
  - run '_wandb login_' on terminal


## For training on stock market data

Change config.py for hyperparameters and artefact storage directories.
Run:
```
python train_stocks.py
```
