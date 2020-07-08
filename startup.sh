#!/bin/bash

pip3 install -r requirements.txt


# Test
# python3 test.py --saved_path trained_models


# Train
python3 train.py --gamma 0.98

