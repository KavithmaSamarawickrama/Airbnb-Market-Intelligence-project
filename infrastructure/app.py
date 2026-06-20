#!/usr/bin/env python3
"""
AWS CDK Application for Airbnb Market Intelligence Pipeline Deployment

This CDK app provisions the complete infrastructure:
- EC2 t3.medium instance (compute)
- S3 Bronze & Silver buckets
- IAM role with scoped permissions
- VPC with auto-stop schedule (cost optimization)
- CloudWatch logs
- Security groups

Deploy with: cdk deploy
"""

import aws_cdk as cdk
from airbnb_pipeline.airbnb_stack import AirbnbPipelineStack

app = cdk.App()

# Load configuration from context
config = {
    "environment": "dev",
    "project_name": "airbnb-market-intel",
    "aws_region": "us-east-1",
    "availability_zone": "us-east-1a",
    "instance_type": "t3.medium",
    "instance_count": 1,
    "tags": {
        "Project": "Airbnb Market Intelligence",
        "Environment": "Development",
        "ManagedBy": "CDK",
        "CostCenter": "DataEngineering",
    },
}

# Create the stack
stack = AirbnbPipelineStack(
    app,
    "AirbnbPipelineStack",
    config=config,
    env=cdk.Environment(region=config["aws_region"]),
)

# Add tags to all resources
for key, value in config["tags"].items():
    cdk.Tags.of(stack).add(key, value)

app.synth()
