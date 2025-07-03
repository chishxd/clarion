#pyright: basic
import io
import pandas as pd
import joblib
import boto3
import os

def get_s3_client(access_key = None, secret_key = None, endpoint_url = None, data_key = None): # type: ignore
    # access_key = "admin"
    # secret_key = "password"
    # endpoint_url = "http://127.0.0.1:9000"
    access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID", "admin")
    secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "root@123")
    endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")


    # --- NEED TO HANDLE EXCEPTION ----
    s3 = boto3.client(  # type: ignore
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key = secret_key, 
        endpoint_url = endpoint_url
    )

    # response = s3.list_buckets()
    # for bucket in response['Buckets']:
    #     print(bucket['Name'])
    return s3  # type: ignore

def df_from_s3(bucket_name: str, object_key: str):
    s3 = get_s3_client() # type: ignore
    response = s3.get_object(Bucket = bucket_name, Key = object_key) # type: ignore
    df = pd.read_csv(io.StringIO(response['Body'].read().decode('utf-8')))  # type: ignore

    return df

def download_file_from_minio(bucket_name: str, object_key: str, local_path: str):
        s3 = get_s3_client()  # type: ignore
        s3.download_file(bucket_name, object_key, local_path)  # type: ignore

def load_model(local_path : str):  # type: ignore
    return joblib.load(local_path)  # type: ignore
