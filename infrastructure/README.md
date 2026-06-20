# AWS CDK Infrastructure as Code

Deploy the complete Airbnb Market Intelligence Pipeline to AWS using AWS Cloud Development Kit.

## 📁 Structure

```
infrastructure/
├── app.py                          # CDK application entry point
├── airbnb_pipeline/
│   ├── __init__.py
│   └── airbnb_stack.py            # Stack definition (EC2, S3, IAM, VPC, etc.)
├── cdk.json                        # CDK configuration
├── requirements.txt                # CDK Python dependencies
├── CDK_DEPLOYMENT_GUIDE.md        # Complete deployment instructions
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- **AWS Account** with credentials configured
- **Node.js** (for AWS CDK CLI)
- **Python 3.11+**

### 2. Install CDK

```bash
npm install -g aws-cdk
cdk --version
```

### 3. Install Dependencies

```bash
cd infrastructure/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Deploy

```bash
# First time only: bootstrap your AWS account
cdk bootstrap aws://YOUR_ACCOUNT_ID/us-east-1

# Review changes
cdk diff

# Deploy!
cdk deploy
```

**That's it!** The stack will be deployed in 5–10 minutes.

## 📋 What Gets Deployed

- **EC2 t3.medium** — Python pipeline execution
- **S3 Bronze Bucket** — Immutable raw data (versioned, encrypted)
- **S3 Silver Bucket** — Processed Parquet data (versioned, encrypted)
- **IAM Role** — Minimal scoped permissions (S3 + CloudWatch only)
- **VPC & Security Group** — Network isolation
- **CloudWatch Logs** — Pipeline execution logs
- **EventBridge Rules** — Auto-stop/start schedule (cost optimization)

## 💰 Cost

- **~$5–10/week** with auto-stop schedule
- **~$30/month** if running 24/7 (without auto-stop)

## 🔐 Security

✅ All S3 buckets are **private** (no public access)  
✅ **Encryption at rest** on S3 and EC2  
✅ **Encryption in transit** (SSL/TLS enforced)  
✅ **IAM scoped permissions** (least privilege)  
✅ **No SSH keys** (use Systems Manager Session Manager)  

## 📖 Full Guide

See **[CDK_DEPLOYMENT_GUIDE.md](CDK_DEPLOYMENT_GUIDE.md)** for:
- Step-by-step deployment
- Post-deployment configuration
- Monitoring & maintenance
- Troubleshooting
- Cost estimation

## 🛠️ Useful Commands

```bash
# Synthesize (generate CloudFormation)
cdk synth

# Review changes before deployment
cdk diff

# Deploy stack
cdk deploy

# Destroy stack (clean up)
cdk destroy

# List stacks
cdk list

# Check logs
cdk logs AirbnbPipelineStack
```

## 🔌 Connect to EC2 Instance

```bash
# SSH via Systems Manager Session Manager (no keys needed)
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=Airbnb*" \
  --region us-east-1 \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

aws ssm start-session --target $INSTANCE_ID --region us-east-1
```

## 📚 References

- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [CDK TypeScript Reference](https://docs.aws.amazon.com/cdk/api/)
- [AWS CloudFormation](https://docs.aws.amazon.com/cloudformation/)
- [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)

---

**See [CDK_DEPLOYMENT_GUIDE.md](CDK_DEPLOYMENT_GUIDE.md) to get started!**
