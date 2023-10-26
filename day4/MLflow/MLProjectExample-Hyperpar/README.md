# Example mlflow project
This repo contains the mlflow project which wraps up all the steps done in the FullExample.ipynb (i.e. the wine classification via ElasticNet) into a pipeline.
More specifically, in the MLProject, four different entrypoints are defined: 
- the download step
- the train/test set split step
- the training step
- the main step, which runs sequentially (via `main.py`) all the entrypoints, handling all the input/output dependencies.
The first three entrypoints explot the papermill library to run a parametrized version of the notebooks. 

To do the correct setup and run the full pipeline:
```
source run_project.sh
```