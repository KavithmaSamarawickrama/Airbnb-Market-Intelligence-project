import boto3
import sys

try:
    session = boto3.Session(profile_name="secondary", region_name="us-east-1")
    s3 = session.client("s3")
    response = s3.list_buckets()
    print("SUCCESS: Listed S3 buckets:")
    for b in response.get('Buckets', []):
        print(f" - {b['Name']}")
except Exception as e:
    print(f"ERROR checking S3: {e}", file=sys.stderr)
