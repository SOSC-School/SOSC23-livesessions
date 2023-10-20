FROM jupyter/scipy-notebook:python-3.10

USER root

RUN wget https://github.com/nats-io/natscli/releases/download/v0.1.1/nats-0.1.1-amd64.deb \
   && dpkg -i nats-0.1.1-amd64.deb && rm  nats-0.1.1-amd64.deb \
   && apt update && apt-get install -y curl

RUN apt-get install -y graphviz

USER jovyan

RUN conda install -y -c conda-forge dask tensorflow

RUN pip3 install boto3 graphviz mimesis black papermill  nats-python pillow tqdm mlflow
RUN mkdir $HOME/data

RUN pip3 install mimesis==8.0.0