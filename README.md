# DDPG_for_finance

## Installation steps

```
conda create -n ddpg_torch_gym python=3.7
conda activate ddpg_torch_gym
pip install -r requirements.txt
```

Install pytorch and cuda based on version:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

weights and biases initialization:
  - run '_wandb login_'


## For training on stock market data

Change config.py for hyperparameters and artefact storage directories.
Run:
```
python train_stocks.py
```
