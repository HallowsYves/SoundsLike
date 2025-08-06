import boto3
import os
import pandas as pd
import json
from io import StringIO, BytesIO
import streamlit as st

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name = st.secrets["AWS_REGION"],
    )

def load_csv_from_s3(file_name, index_col=None):
    s3 = get_s3_client()
    bucket = st.secrets["S3_BUCKET_NAME"]
    folder = st.secrets["S3_FOLDER"]
    key = f"{folder}{file_name}"

    response = s3.get_object(Bucket=bucket, Key=key)
    csv_data = response["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(csv_data), index_col=index_col)

def load_json_from_s3(file_name):
    s3 = get_s3_client()
    bucket = st.secrets["S3_BUCKET_NAME"]
    folder = st.secrets["S3_FOLDER"]
    key = f"{folder}{file_name}"

    response = s3.get_object(Bucket=bucket, Key=key)
    json_data = response["Body"].read().decode("utf-8")
    return json.loads(json_data)

def load_binary_from_s3(file_name):
    s3 = get_s3_client()
    bucket = st.secrets["S3_BUCKET_NAME"]
    folder = st.secrets["S3_FOLDER"]
    key = f"{folder}{file_name}"

    response = s3.get_object(Bucket=bucket, Key=key)
    return BytesIO(response["Body"].read())
