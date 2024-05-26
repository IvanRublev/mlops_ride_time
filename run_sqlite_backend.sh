#!/bin/sh
echo """
In the parallel terminal, pelase, run the following:
python hpo.py --data_path inputs
python register_model.py --data_path inputs
"""

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts 
