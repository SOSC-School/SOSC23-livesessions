## Quick start

Change username and passwords in `.env` files as you wish.

Then just drop:

```bash
docker compose up -d
```

on `localhost:8888` you should access the jupyterlab instance with the token you put in `.env` file. Same for `localhost:40093` where you should be able to see minio interface and on `localhost:5000` the MLFlow UI.
