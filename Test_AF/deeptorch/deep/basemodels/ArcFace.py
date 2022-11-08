from deep.basemodels.arcface_torch.backbones import get_model
import torch

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


# def loadModel(weight='/Test_AF/deeptorch/deep/basemodels/arcface_torch/weights/ms1mv3_arcface_r100_fp16/backbone.pth', name='r100', eval=True):
def loadModel(weight='/Test_AF/deeptorch/deep/basemodels/arcface_torch/weights/ms1mv3_arcface_r100_fp16/backbone.pth', name='r100', eval=True):
	
	# download_files(
    # s3_client,
    # "rag-net-v2-0c6f96b8050c43fd-inputs",
    # "/app/weights",
    # file_names='backbone.pth',
	# dir="arc"
	# )

	fil = 'input/arc/backbone.pth'

	s3_client.download_file(
		"rag-net-v2-0c6f96b8050c43fd-inputs",
		fil,
		"/app/weights/backbone.pth"
	)

	weight = "/app/weights/backbone.pth"

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	model = get_model(name, fp16=False)
	
	model.load_state_dict(torch.load(weight))
	print("Weights are loaded")
	
	if eval == True:
		model.eval()
		model.to(device)
	return model
