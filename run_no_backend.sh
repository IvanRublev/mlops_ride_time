#!/bin/sh
python preprocess_data.py --raw_data_path data --dest_path inputs
python train.py --data_path inputs
mlflow ui
