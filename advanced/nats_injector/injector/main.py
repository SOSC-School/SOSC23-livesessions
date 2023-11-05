# Python Standard Libraries 
import os
import tarfile 
import io
import logging 
from glob import glob
import asyncio
from datetime import datetime, timedelta
import json
from hashlib import md5
import sqlite3 as sql

## Numerical third-party modules
import numpy as np      ## Numerical computations on the image
from PIL import Image   ## Handlers for image file formats (JPG, PNG, ...)

## Standard third-party modules
import requests  ## HTTP(s)  
import minio     ## MinIO SDK 
import nats      ## Messaging system

## Others
from tqdm import tqdm   ## Progress bar 


################################################################################
## Configuration 
## -------------
## The following variables must, should or can be defined as environmental 
## variables, ideally in the docker-compose.
##

## MinIO configuration
MINIO_URL = os.environ.get("MINIO_URL", "minio:9000")
MINIO_USER = os.environ.get("MINIO_USER", '3TVdK/YmR1nc!pDpS?doTn8qT0gNiFO4cj8Vhdo8')
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", 'DgloAlKQz55EVogiqOd6KvCgtTxVs1njs-TvheufwIt8Cjw9kNtBz73n2R6AY9rl')
BUCKET_NAME = os.environ.get("BUCKET_NAME", "cygno-daq")

## NATS server 
NATS_SERVER = os.environ.get("NATS_SERVER", "nats://localhost:4222")

## Remote data repository
HTTP_DATA_SOURCE = os.environ.get("HTTP_DATA_SOURCE", "https://pandora.infn.it/public/16cf05/dl/cygno-sim.tar")

## Logging format for interactive development and log persistency
LOGGING_FORMAT = os.environ.get("LOGGING_FORMAT", "%(asctime)s - %(message)s")

## Cache configuration
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp")
ORIGINAL_DATASET_CACHE = os.environ.get("ORIGINAL_DATASET_CACHE", os.path.join(CACHE_DIR, "original"))

## Signal and background configuration
## The probability of picking a signal event is defined as 
##   MAX_SIGNAL_FRACTION * (1 + sin(2 * pi * t[h] / PERIOD_HOURS)
MAX_SIGNAL_FRACTION = float(os.environ.get("MAX_SIGNAL_FRACTION", "0.1"))
PERIOD_HOURS = float(os.environ.get("PERIOD_HOURS", "3"))
## Start date for the SOSC, defines the phase of the oscillation
START = datetime(year=2023, month=10, day=23)
## While the total number of images submitted per hour is set by EVENTS_PER_HOUR
EVENTS_PER_HOUR = float(os.environ.get("EVENTS_PER_HOUR", "300"))

## The images (both signal and backround) are corrupted with additional noise and 
## JPEG compression. NOISE_LEVEL is the RMS of a Gaussian noise superposed to the original image.
NOISE_LEVEL = float(os.environ.get("NOISE_LEVEL", "0.02"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

## Database where the truth (whether signal or background) of each daq is stored
SQLITE_TRUTH_DB = os.environ.get("SQLITE_TRUTH", "truth.db")


def _download_file(url, local_filename=None):
    """
    Download a file streaming it to a local file to avoid OOM errors.
    """
    local_filename = local_filename or url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tqdm(total=2300) as progress_bar:   
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=64<<20):
                    progress_bar.update(64)
                    f.write(chunk)

    return local_filename



def ensure_bucket(minio_client, bucket_name=BUCKET_NAME):
    """
    Ensure the bucket `bucket_name` is available.

    """
    if not minio_client.bucket_exists(bucket_name):
        logging.debug(f"Bucket {bucket_name} not found. Creating it")
        minio_client.make_bucket(bucket_name)




async def ensure_local_dataset(nats_client, get_tar_url=None):
    """
    Ensure a local copy of the dataset exists, downloading it from get_tar_url if necessary.

    if get_tar_url is None, the environment variable "HTTP_DATA_SOURCE" is used.
    """
    if get_tar_url is None:
        get_tar_url = HTTP_DATA_SOURCE
    
    if get_tar_url is None:
        raise ValueError("HTTP_DATA_SOURCE not set")

    n_objects = len(glob(os.path.join(ORIGINAL_DATASET_CACHE, "*/*.png")))
    await log_message(nats_client, f"Found {n_objects} in ./original/")
    if n_objects < 1000:
        if not os.path.exists(ORIGINAL_DATASET_CACHE):
            await log_message(
                    nats_client, 
                    f"Insufficient number of files. Downloading original dataset."
                    )

            original_dataset = os.path.join(CACHE_DIR, "original.tar")
            if not os.path.exists(original_dataset):
                _download_file(get_tar_url, original_dataset)
                logging.debug(f"Downloaded to {original_dataset}")

            archive = tarfile.open(original_dataset)
            archive.extractall(ORIGINAL_DATASET_CACHE)
            archive.close()

            os.remove(original_dataset)
    
    nr_files = glob(os.path.join(ORIGINAL_DATASET_CACHE, "NR/*.png"))
    er_files = glob(os.path.join(ORIGINAL_DATASET_CACHE, "ER/*.png"))

    return nr_files, er_files


def _get_hours_since_start():
    """
    Number of hours (as a floating point) since START
    """
    return (datetime.now() - START).total_seconds()/(60*60)


async def _pick_smear_compress(nats_client, minio_client, signal, background):
    """
    Pick a random figure, add noise, compress it in JPEG format, upload on minio and publish url on nats
    """
    t = _get_hours_since_start()
    probability_signal = np.clip(
            MAX_SIGNAL_FRACTION * (1 + np.sin(2 * np.pi * t / PERIOD_HOURS))/2,
            0, 1)

    is_signal = np.random.uniform(0, 1) < probability_signal
    filename = np.random.choice(signal if is_signal else background)

    image = np.array(Image.open(filename)).astype(np.float64)/255
    image += np.random.normal(0., NOISE_LEVEL, image.shape)
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    image_buff = io.BytesIO()
    Image.fromarray(image).save(image_buff, quality=JPEG_QUALITY, format="jpeg")
    # Image.fromarray(image).save(image_buff, format="png")

    md5sum = md5(image_buff.getbuffer())
    object_name = f"cygno-{md5sum.hexdigest()}.jpg"

    image_buff.seek(0)
    minio_client.put_object(BUCKET_NAME, object_name, image_buff, length=len(image_buff.getbuffer()))

    timestamp = str(datetime.now())

    with sql.connect(SQLITE_TRUTH_DB) as db:
        db.execute("CREATE TABLE IF NOT EXISTS truth (tstamp, original, filename, is_signal);");
        db.execute("INSERT INTO truth (tstamp, original, filename, is_signal) VALUES (?, ?, ?, ?)",
                (timestamp, filename, object_name, 1 if is_signal else 0)
                );

    url = minio_client.presigned_get_object(BUCKET_NAME, object_name, expires=timedelta(hours=1))

    await nats_client.publish("daq/data", 
            json.dumps(dict(
                url=url, 
                time=timestamp, 
                filename=object_name,
              )).encode('utf-8')
        )

    await log_message(nats_client, f"Event {object_name} uploaded to {BUCKET_NAME} and submitted to 'daq/data'")



def log_stats():
    with sql.connect(SQLITE_TRUTH_DB) as db:
        for row in db.execute("SELECT tstamp, filename, is_signal FROM truth"):
            print (row)


async def log_message(nats_client, msg: str):
    """
    Writes a message to stdout AND to NATS with subject "daq/info"
    """
    await nats_client.publish("daq/info", msg.encode("ascii"))
    logging.info(f"{datetime.now()}  {msg}")
    
async def main():
    """
    Application loop.
    """
    nc = await nats.connect(NATS_SERVER)
    mc = minio.Minio(MINIO_URL, MINIO_USER, MINIO_ACCESS_KEY, secure=False)

    await log_message(nc, "Nats injector ready.")
    ensure_bucket(mc)
    await log_message(nc, 'Target bucket exists.')
    nr_files, er_files = await ensure_local_dataset(nc)
    await log_message(nc, 'Local dataset is now available.')

    while True:
        await _pick_smear_compress(nc, mc, nr_files, er_files)
        deltat = -np.log(np.random.uniform(0, 1))/EVENTS_PER_HOUR * 3600
        print (f"Next event at {datetime.now() + timedelta(seconds=deltat)}.")
        await asyncio.sleep(deltat)

    log_stats()




if __name__ == '__main__':
    logging.basicConfig(
            level=logging.DEBUG,
            format=LOGGING_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            )
    logging.info("Server started")
    asyncio.run(main())
