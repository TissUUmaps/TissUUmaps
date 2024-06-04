"""
Methods to parse the .geojson files and push them to S3

Developed by Mena S.A. Kamel (mena.sa.kamel@gmail.com | mena.kamel@sanofi.com)
date: June 4, 2024

Requirements: Set environment variables for AWS credentials:
    - AWS_ACCESS_KEY: AWS key ID
    - AWS_SECRET_KEY: AWS secret access key
"""
import os
import boto3
import json
from io import BytesIO

class Plugin:
    """Class to handle all the backend methods for the Annotation_S3_Upload plugin"""

    def __init__(self, app):
        """Initializes the plugin"""
        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
        )

    def upload_annotations(self, jsonParam):
        """
        This function uploads the geojsons specified in jsonParam["geojsons"]
        to S3
        """
        geojsons = jsonParam["geojsons"]
        bucket_name = jsonParam["bucket_name"]
        file_location = jsonParam["file_location"]
        geojson_bytes = json.dumps(geojsons).encode("utf-8")
        memory_file = BytesIO(geojson_bytes)
        self.s3_client.upload_fileobj(memory_file, bucket_name, file_location)
        return {"url": file_location}