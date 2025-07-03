"""
Utility functions for interacting with S3.
"""
#pyright: basic
import io
import pandas as pd
import joblib
import boto3
import os

def get_s3_client(access_key = None, secret_key = None, endpoint_url = None, data_key = None): # type: ignore
    """
    Initializes and returns an S3 client.

    Credentials can be passed as arguments or set as environment variables.
    Falls back to default credentials if none are provided.

    Args:
        access_key (str, optional): AWS access key ID. Defaults to None.
        secret_key (str, optional): AWS secret access key. Defaults to None.
        endpoint_url (str, optional): S3 endpoint URL. Defaults to None.
        data_key (None, optional): This parameter is not used. Defaults to None.

    Returns:
        boto3.client: An S3 client object.
    """
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
    """
    Reads a CSV file from an S3 bucket and returns it as a pandas DataFrame.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_key (str): The key of the object (file) in the bucket.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    s3 = get_s3_client() # type: ignore
    response = s3.get_object(Bucket = bucket_name, Key = object_key) # type: ignore
    df = pd.read_csv(io.StringIO(response['Body'].read().decode('utf-8')))  # type: ignore

    return df

def download_file_from_minio(bucket_name: str, object_key: str, local_path: str):
        """
        Downloads a file from an S3-compatible service (like MinIO) to a local path.

        Args:
            bucket_name (str): The name of the bucket.
            object_key (str): The key of the object to download.
            local_path (str): The local path to save the downloaded file.
        """
        s3 = get_s3_client()  # type: ignore
        s3.download_file(bucket_name, object_key, local_path)  # type: ignore

def load_model(local_path : str):  # type: ignore
    """
    Loads a model from a local file using joblib.

    Args:
        local_path (str): The path to the model file.

    Returns:
        Any: The loaded model object.
    """
    return joblib.load(local_path)  # type: ignore
