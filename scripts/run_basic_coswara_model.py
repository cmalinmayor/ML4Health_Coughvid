# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from coughvid.train_model import ModelConfig, load_model, get_dataloaders, train_model

# +
data_dir = './data/coswara/'
metadata_file = 'filtered_data.csv'
batch_size = 1
num_workers = 1
model_dir = 'trained_models'
logging_dir = 'logs'
os.makedirs(model_dir, exist_ok=True)

model_config = ModelConfig(data_dir, batch_size, num_workers, model_dir=model_dir, logging_dir=logging_dir):
  

# +
leaf=False

dataloaders = get_dataloaders(model_config, leaf=leaf)

# +
model_type='resnet18'
num_epochs=50
use_wandb=False

train_model(model_config, dataloaders, model_type=model_type, num_epochs=num_epochs, use_wandb=use_wandb)
