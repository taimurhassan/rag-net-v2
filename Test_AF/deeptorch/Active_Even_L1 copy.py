import pandas as pd
import itertools
from tqdm import tqdm
from os.path import expanduser
from pathlib import Path
import pickle
import os
import json
from deep import DeepFace
import boto3

def download_files(s3_client, bucket_name, local_path, file_names, dir):

    local_path = Path(local_path)

    for x in file_names:
        file_path = Path.joinpath(local_path, x)
        print(file_path, x)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(file_path)
        s3_client.download_file(
            bucket_name,
            dir + x,
            str(file_path)
        )

model_name = "PocketNet"
distance_metric = "euclidean_l2"
detector_backend = 'opencv'

AWS_S3_INPUT_BUCKET = os.getenv("AWS_S3_INPUT_BUCKET")
AWS_S3_OUTPUT_BUCKET = os.getenv("AWS_S3_OUTPUT_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID")

session = boto3.session.Session()

s3_client = session.client(
    service_name='s3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=S3_ENDPOINT,
)

#P1
idendities = {
    "1": ["down.jpg"],
    "2": ['frame463.jpg']
    }

id = ["down.jpg", 'frame463.jpg']


positives = []

for key, values in idendities.items():
    
    #print(key)
    for i in range(0, len(values)-1):
        for j in range(i+1, len(values)):
            #print(values[i], " and ", values[j])
            positive = []
            positive.append(values[i])
            positive.append(values[j])
            positives.append(positive)

positives = pd.DataFrame(positives, columns = ["file_x", "file_y"])
positives["decision"] = "Yes"

samples_list = list(idendities.values())


negatives = []

for i in range(0, len(idendities) - 1):
    for j in range(i+1, len(idendities)):
        #print(samples_list[i], " vs ",samples_list[j]) 
        cross_product = itertools.product(samples_list[i], samples_list[j])
        cross_product = list(cross_product)
        #print(cross_product)
        
        for cross_sample in cross_product:
            #print(cross_sample[0], " vs ", cross_sample[1])
            negative = []
            negative.append(cross_sample[0])
            negative.append(cross_sample[1])
            negatives.append(negative)

negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
negatives["decision"] = "No"

df = pd.concat([positives, negatives]).reset_index(drop = True)

negatives["decision"] = "Yes"

negatives.decision.value_counts()

download_files(
    s3_client,
    "rag-net-v2-0c6f96b8050c43fd-inputs",
    "/app/input",
    file_names=id,
    dir="droneSURF/Active_Even_L1/1/"
)

dataset_path = "/app/input/"


negatives.file_x = dataset_path + negatives.file_x
negatives.file_y = dataset_path + negatives.file_y


instances = negatives[["file_x", "file_y"]].values.tolist()


print("P1")
resp_obj1 = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, enforce_detection = False, detector_backend = detector_backend)
print("P1 Done")

os.mkdir("/app/output")


fil = 'output/PocketNetActiveEvenL1/di.json'
s3_client.download_file(
    "rag-net-v2-0c6f96b8050c43fd-outputs",
    fil,
    "/app/output/di.json"
)

with open("/app/output/di.json", "w") as outfile:
    json.dump(resp_obj1, outfile)

s3_client.upload_file("/app/output/di.json", 
"rag-net-v2-0c6f96b8050c43fd-outputs", 
"output/PocketNetActiveEvenL1/di.json")