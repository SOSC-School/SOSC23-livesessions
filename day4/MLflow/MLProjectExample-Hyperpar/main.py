import mlflow
import click
import os
import sys
import numpy as np


@click.command()
@click.option("--data-url", default="tt", type=str)
def workflow(data_url):
    download_data_run = mlflow.run(".", "download_data", parameters={"data_url": data_url}, env_manager="local") 
    download_data_run.wait()
    artifact_path = mlflow.get_run(download_data_run.run_id).info.artifact_uri
    prepare_train_test_run = mlflow.run(".", "prepare_train_test", parameters={"csv_path": "/".join([artifact_path, 'dataset.csv'])}, env_manager="local") 
    prepare_train_test_run.wait()
    artifact_path = mlflow.get_run(prepare_train_test_run.run_id).info.artifact_uri
    best_run_id = 00
    best_metric = -99999999
    for alpha_ in np.linspace(0.,1., num=3):
        for l1_ratio_ in np.linspace(0.,1., num = 3):
            train_run = mlflow.run(".", "train", parameters={"csv_train_path": "/".join([artifact_path, 'train.csv']), "csv_test_path": "/".join([artifact_path, 'test.csv']), "alpha": alpha_, "l1_ratio": l1_ratio_}, env_manager="local") 
            train_run.wait()
            print()
            metric = mlflow.get_run(train_run.run_id).data.to_dictionary()['metrics']['rmse']
            #print(metric)
            #print(train_run.run_id)
            #print()
            if metric > best_metric:
                best_metric = metric
                best_run_id = train_run.run_id
    #train_run = mlflow.run(".", "train", parameters={"csv_train_path": "/".join([artifact_path, 'train.csv']), "csv_test_path": "/".join([artifact_path, 'test.csv']), "alpha": 0.5, "l1_ratio": 0.5}, env_manager="local")
    #train_run.wait()
    print("best")
    print(best_metric)
    print(best_run_id)
    artifact_path = mlflow.get_run(best_run_id).info.artifact_uri
    test_run = mlflow.run(".", "test", 
                        parameters={"model_path": "/".join([artifact_path, '/model'])}, 
                           env_manager="local"
                          )
    
    
if __name__ == "__main__":
    workflow()