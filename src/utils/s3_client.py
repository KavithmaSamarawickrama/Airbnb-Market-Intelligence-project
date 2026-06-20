"""
Utility module for S3 client initialization and operations.

Handles credentials, IAM role detection, bucket validation, and common S3 operations.
"""

import logging
import boto3
from typing import Dict, Optional, Tuple
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3Client:
    """Wrapper around boto3 S3 client with defensive error handling."""

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """
        Initialize S3 client.

        Args:
            region: AWS region (default: us-east-1)
            profile: AWS profile name (if None, uses default credentials)
        """
        try:
            if profile:
                session = boto3.Session(profile_name=profile, region_name=region)
                self.client = session.client("s3")
                self.session = session
            else:
                self.client = boto3.client("s3", region_name=region)
                self.session = boto3.Session(region_name=region)
            logger.info(f"S3 client initialized for region {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Check ~/.aws/credentials or IAM role.")
            raise

    def head_bucket(self, bucket: str) -> bool:
        """Check if bucket exists and is accessible."""
        try:
            self.client.head_bucket(Bucket=bucket)
            logger.debug(f"Bucket {bucket} exists and is accessible")
            return True
        except ClientError as e:
            status_code = e.response.get("Error", {}).get("Code")
            logger.warning(f"Bucket {bucket} check failed: {status_code}")
            return False

    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
        metadata: Optional[Dict[str, str]] = None,
        sse: str = "AES256",
    ) -> bool:
        """
        Upload a single object to S3.

        Args:
            bucket: S3 bucket name
            key: Object key
            body: Binary content
            metadata: Optional metadata dict
            sse: Server-side encryption (AES256, aws:kms, etc.)

        Returns:
            True if successful, False otherwise
        """
        try:
            extra_args = {"ServerSideEncryption": sse}
            if metadata:
                extra_args["Metadata"] = metadata

            self.client.put_object(Bucket=bucket, Key=key, Body=body, **extra_args)
            logger.debug(f"Uploaded {key} to {bucket}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {key} to {bucket}: {e}")
            return False

    def get_object(self, bucket: str, key: str) -> Optional[bytes]:
        """
        Download an object from S3.

        Args:
            bucket: S3 bucket name
            key: Object key

        Returns:
            Binary content if successful, None otherwise
        """
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            logger.debug(f"Downloaded {key} from {bucket}")
            return content
        except ClientError as e:
            logger.warning(f"Failed to download {key} from {bucket}: {e}")
            return None

    def list_objects(
        self, bucket: str, prefix: str = "", max_keys: int = 1000
    ) -> list:
        """
        List objects in a bucket with a given prefix.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix filter
            max_keys: Max results (default 1000)

        Returns:
            List of object dicts (Key, Size, LastModified, etc.)
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, MaxKeys=max_keys
            )
            objects = response.get("Contents", [])
            logger.debug(f"Listed {len(objects)} objects from {bucket}/{prefix}")
            return objects
        except ClientError as e:
            logger.error(f"Failed to list objects in {bucket}/{prefix}: {e}")
            return []

    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete an object from S3."""
        try:
            self.client.delete_object(Bucket=bucket, Key=key)
            logger.debug(f"Deleted {key} from {bucket}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete {key} from {bucket}: {e}")
            return False

    def multipart_upload(
        self, bucket: str, key: str, file_path: Path, sse: str = "AES256"
    ) -> bool:
        """
        Upload a large file using multipart upload.

        Args:
            bucket: S3 bucket name
            key: Object key
            file_path: Local file path
            sse: Server-side encryption

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.upload_file(
                str(file_path),
                bucket,
                key,
                ExtraArgs={"ServerSideEncryption": sse},
            )
            logger.info(f"Multipart uploaded {file_path} to {bucket}/{key}")
            return True
        except ClientError as e:
            logger.error(f"Multipart upload failed: {e}")
            return False
