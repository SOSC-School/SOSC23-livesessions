####
## This Dockerfile defines an image for JupyterHub capable of 
## running docker, via Docker-in-docker (dind). 
## This enables the creation of interactive notebooks explaining docker.
##
## It is based on Ubuntu jammy. This requires repackaging the official 
## dind image which is based on alpine, instead. 
## Unfortunately, installing tensorflow on Alpine Linux is a nightmare.
## Fortunately, the problem has been solved by Carlos Ruiz Ballesteros
## and we picked his Docker image.
## 

## See: https://github.com/cruizba/ubuntu-dind
FROM cruizba/ubuntu-dind:jammy-24.0.6

## Libraries
RUN apt-get update && apt-get install -qy \
       vim \
       linux-headers-generic gcc g++ make \
       libffi-dev \
       git \
       bash \
       tini \
       python3.10 \
       unzip \
       htop \
       && apt-get clean

## Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

## Install wheel (prerequisite to install binaries with pip)
RUN pip install wheel

## Install generic ipython/jupyter dependencies 
RUN pip install --no-cache-dir \
  numpy pandas pillow matplotlib dask jupyterlab requests \
  ipython jupyter notebook==6.* jupyterlab jupyterhub==1.5.0 \
  docker tqdm h5py jupyter-server-proxy dask-labextension \
  nats-python minio emnist

## Install tensorflow on CPU
RUN pip install tensorflow

## Configure Jupyter based on DODAS specifications
RUN mkdir /usr/local/share/dodasts
WORKDIR /usr/local/share/dodasts
RUN git clone https://github.com/DODAS-TS/jupyterlab-collaborative-util-extension.git
WORKDIR /usr/local/share/dodasts/jupyterlab-collaborative-util-extension
#     sed -i '/"compilerOptions": {/a \    "lib": ["es2018", "dom"],' tsconfig.json && \
#     jupyter labextension install
WORKDIR /usr/local/share/dodasts

## 
ARG DODAS_GITHUB=https://raw.githubusercontent.com/DODAS-TS/dodas-docker-images/master
RUN mkdir -p /usr/local/share/jupyter/ && \
    wget ${DODAS_GITHUB}/docker/single-node-jupyterhub/lab/jupyter_lab_config.py \
    -O /usr/local/share/jupyter/jupyter_lab_config.py

## Install node.js and npm from NodeSource
ARG NODE_VER=20
RUN cd /root && \
    apt-get update && \
    apt-get install -y ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_VER}.x nodistro main"\
    | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install nodejs -y && apt-get clean


## Build jupyter lab stuff
RUN /bin/bash -c "jupyter lab build --dev-build=False --minimize=True"

## Configure the script to be executed after spawning. 
## N.B. JupyterHub MUST BE CONFIGURED to run this script, otherwise it will be ignored.
ARG POSTSCRIPT=/usr/local/share/dodasts/script/post_script.sh
RUN mkdir /usr/local/share/dodasts/script && \
    echo "#!/bin/bash " > $POSTSCRIPT && \
    echo "dockerd-entrypoint.sh &> /var/log/dockerd &" >> $POSTSCRIPT && \
    chmod +x $POSTSCRIPT && \
    echo 'DOCKER_HOST=""' >> /etc/environment

## Create the user directory
RUN mkdir /jupyter-workspace
WORKDIR /jupyter-workspace

## Fake command, necessary to JupyterHub, but replaced in configuration.
CMD ["/bin/bash"]

