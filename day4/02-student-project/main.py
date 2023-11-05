import mlflow
import click
import os
import sys
import warnings

warnings.filterwarnings('ignore')

@click.command()
@click.option("--batch_size", default=10, type=int)
@click.option("--n_epochs", default=5, type=int)
def workflow(batch_size, n_epochs):

    train = mlflow.run(".", "train", env_manager="local", parameters={"batch_size": batch_size, "n_epochs": n_epochs}) 

    train.wait()

    model_run_uri = mlflow.get_run(train.run_id).info.artifact_uri

    evaluate = mlflow.run(".", "evaluate", parameters={"model_run_uri": "/".join([model_run_uri, 'classifier.keras'])}, env_manager="local") 

    evaluate.wait()

if __name__ == "__main__":
    workflow()