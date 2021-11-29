#!/bin/bash

srun --partition=biggpunodes -c 4 --gres=gpu:1 python coswara_basic_model.py 
