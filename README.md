# DDPG_for_finance

## Installation steps

```
conda create -n ddpg_trading python=3.8
conda activate ddpg_trading
pip install -r requirements.txt
```

Install pytorch based on system from [here](https://pytorch.org/get-started/locally/):
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Make sure you have an active weights and biases account login:
  - run '_wandb login_' in terminal

## For hyperparameter tuning before training [Optional]
- Make changes in "config_tuning.py" if you want to change num_trials etc.
- Check train test split in hyperparemeter_tuning.py script and
Run:
```
python hyperparemeter_tuning.py
```
Hyperparameters returned can then be updated in _config.py_ before running training in next step.

## For training on stock market data [without retraining]

- Change config.py for hyperparameters, artefact storage directories, wandb project name etc.

Run:
```
python train_ddpg_agent.py
```

## For training on stock market data [with retraining]

- Change config.py for hyperparameters, artefact storage directories, wandb project name etc.

Run:
```
python train_ddpg_agent_with_retraining.py
```
Note: if the parameter _'do_hyp_tuning'_ is set to True in _train_ddpg_agent_with_retraining.py_, you don't need to mention hyperparams in config as they won't be used. For every training, optimal hyperparameters will be found and used.