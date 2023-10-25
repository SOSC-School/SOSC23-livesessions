## MLFlow

Create an MLFlow experiment
```
export MLFLOW_TRACKING_INSECURE_TLS=true
export MLFLOW_TRACKING_URI=https://${JUPYTERHUB_USER}-mlflow.131.154.99.220.myip.cloud.infn.it/
mlflow experiments create -n experiment1
```

Run the training and evaluate steps:
```
mlflow run --env-manager=local ./ --experiment-name experiment1
```

### Tracking server

```
https://${JUPYTERHUB_USER}-mlflow.131.154.99.220.myip.cloud.infn.it/
```