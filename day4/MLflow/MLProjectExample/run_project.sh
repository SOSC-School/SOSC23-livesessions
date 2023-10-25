export MLFLOW_TRACKING_INSECURE_TLS=true
export MLFLOW_TRACKING_URI=https://${JUPYTERHUB_USER}-mlflow.131.154.99.220.myip.cloud.infn.it/
mlflow experiments create -n myFirstFullPipeline
mlflow run ./ --env-manager=local --experiment-name myFirstFullPipeline --run-name Main