import boto3
import os
import streamlit as st

"""
We'd use this adter connecting it to our S3 bucket:

from s3_utils import read_csv_from_s3
df = read_csv_from_s3("data/my_dataset.csv")

Now there's some things to do in the .streamlit/secrets.toml
"""
def get_s3_client():
    if st.secrets.get("aws"):
        # Running in Streamlit Cloud
        aws_secrets = st.secrets["aws"]
    else:
        # Local .env
        from dotenv import load_dotenv
        load_dotenv()
        aws_secrets = {
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "AWS_REGION": os.getenv("AWS_REGION"),
            "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME"),
        }

    return boto3.client(
        "s3",
        aws_access_key_id=aws_secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_secrets["AWS_REGION"],
    ), aws_secrets["S3_BUCKET_NAME"]

def read_csv_from_s3(key):
    s3, bucket = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    import pandas as pd
    return pd.read_csv(response['Body'])
