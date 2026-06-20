# AWS CDK Deployment Guide — Airbnb Market Intelligence Pipeline

**Purpose:** Deploy the complete infrastructure to AWS (EC2, S3, IAM, CloudWatch, auto-scaling)  
**Estimated Duration:** 15–20 minutes  
**Cost:** ~$5–10/week (EC2 t3.medium + S3 storage)

---

## 📋 Prerequisites

### 1. Install AWS CDK CLI

```bash
# Install Node.js (required for CDK CLI)
# Download from: https://nodejs.org/ (LTS version)

# Install AWS CDK CLI globally
npm install -g aws-cdk

# Verify installation
cdk --version
```

**Expected output:** `X.XX.X (build XXXXX)`

### 2. Configure AWS Credentials

```bash
# Option 1: Using AWS CLI
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Default region (us-east-1)

# Option 2: Using environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Verify credentials
aws sts get-caller-identity
```

**Expected output:**
```json
{
    "UserId": "AIDAI...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/username"
}
```

### 3. Install CDK Dependencies

```bash
cd infrastructure/
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

---

## 🚀 Deployment Steps

### Step 1: Bootstrap AWS Account (First-Time Only)

Before deploying any CDK stacks, bootstrap your AWS account:

```bash
cdk bootstrap aws://123456789012/us-east-1
```

**What it does:**
- Creates S3 bucket for CDK artifacts
- Creates IAM roles for CloudFormation
- One-time setup per account/region

**Expected output:**
```
 ✓ Environment aws://123456789012/us-east-1 bootstrapped.
```

### Step 2: Synthesize CloudFormation Template

Generate the CloudFormation template from CDK code:

```bash
cdk synth
```

**What it does:**
- Generates `cdk.out/AirbnbPipelineStack.template.json`
- Validates CDK code

**Expected output:**
```
Successfully synthesized to cdk.out/AirbnbPipelineStack.template.json
```

### Step 3: Review Deployment Plan (Diff)

Before deploying, review what resources will be created:

```bash
cdk diff
```

**What you'll see:**
- List of resources to be created (EC2, S3 buckets, IAM roles, etc.)
- Changes to security groups
- IAM permission grants

**Example output:**
```
[+] AWS::EC2::Instance airbnb-market-intel-instance ...
[+] AWS::S3::Bucket airbnb-market-intel-bronze-123456789012 ...
[+] AWS::S3::Bucket airbnb-market-intel-silver-123456789012 ...
[+] AWS::IAM::Role airbnb-market-intel-ec2-role ...
...
```

### Step 4: Deploy Stack

Deploy the infrastructure to AWS:

```bash
cdk deploy
```

**What happens:**
1. Uploads CDK code to S3
2. Creates CloudFormation stack
3. Provisions all resources in order:
   - VPC & Security Group
   - IAM roles & policies
   - S3 buckets with encryption
   - EC2 instance (with bootstrap script)
   - CloudWatch logs
   - EventBridge auto-stop/start schedules

**Expected time:** 5–10 minutes  
**Expected output:**
```
AirbnbPipelineStack: creating CloudFormation changeset...
AirbnbPipelineStack: 12 resources created
AirbnbPipelineStack: done

Outputs:
AirbnbPipelineStackInstancePublicIp = 54.123.45.67
AirbnbPipelineStackBronzeBucketName = airbnb-market-intel-bronze-123456789012
AirbnbPipelineStackSilverBucketName = airbnb-market-intel-silver-123456789012
```

### Step 5: Verify Deployment

```bash
# List stacks
aws cloudformation list-stacks --region us-east-1

# Describe EC2 instance
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=Airbnb*" \
  --region us-east-1 \
  --query 'Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress]' \
  --output text
```

**Expected output:**
```
i-0abc1234def567890  running  54.123.45.67
```

---

## 🔗 Post-Deployment Steps

### Step 1: Connect to EC2 Instance

Use AWS Systems Manager Session Manager (no SSH keys needed):

```bash
# Find instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=Airbnb*" \
  --region us-east-1 \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

# Start session
aws ssm start-session --target $INSTANCE_ID --region us-east-1
```

**Inside the session:**
```bash
cd /opt/airbnb-pipeline

# Check that repository was cloned
ls -la

# Verify Python environment
source venv/bin/activate
python --version

# Check cron job
crontab -l

# Manual test: run ingestion
python src/ingest.py

# View logs
tail -f /var/log/airbnb-pipeline.log
```

### Step 2: Update S3 Bucket Names in Config

Update `config/settings.yaml` with the CDK-generated bucket names:

```bash
# Get bucket names
aws s3 ls | grep airbnb-market-intel

# Update config/settings.yaml
# Replace:
# s3:
#   bronze_bucket: "airbnb-intel-bronze-{{ account_id }}"
# With:
# s3:
#   bronze_bucket: "airbnb-market-intel-bronze-123456789012"
```

### Step 3: Set Up Auto-Stop Schedule

Verify EventBridge rules are active:

```bash
aws events list-rules --name-prefix airbnb --region us-east-1

# Expected: two rules
# - airbnb-market-intel-auto-stop-rule (8 PM UTC)
# - airbnb-market-intel-auto-start-rule (8 AM UTC, weekdays only)
```

**Cost Impact:**
- **Without auto-stop:** $29/month (24/7 running t3.medium)
- **With auto-stop:** $3–5/month (8 hours/day, weekdays only)

### Step 4: Monitor CloudWatch Logs

View pipeline execution logs:

```bash
# List recent logs
aws logs tail /airbnb/pipeline/execution --follow --region us-east-1

# Or view in AWS Console:
# CloudWatch → Log Groups → /airbnb/pipeline/execution
```

---

## 🛠️ Management Tasks

### View Stack Status

```bash
# Describe stack
aws cloudformation describe-stacks \
  --stack-name AirbnbPipelineStack \
  --region us-east-1
```

### Update Stack (After Code Changes)

```bash
# Modify CDK code in airbnb_pipeline/airbnb_stack.py

# Review changes
cdk diff

# Deploy updates
cdk deploy
```

### Destroy Stack (Clean Up)

```bash
# WARNING: This deletes all resources
cdk destroy

# Confirm deletion
# Type: y
```

**Note:** S3 buckets with data will fail deletion. To force delete:

```bash
# Empty S3 buckets first
aws s3 rm s3://airbnb-market-intel-bronze-123456789012 --recursive
aws s3 rm s3://airbnb-market-intel-silver-123456789012 --recursive

# Then destroy
cdk destroy
```

### Scale EC2 Instance Type

To change instance type (e.g., t3.small to t3.medium):

1. Edit `cdk.json`:
   ```json
   {
     "context": {
       "instance_type": "t3.large"  // Change this
     }
   }
   ```

2. Deploy update:
   ```bash
   cdk deploy
   ```

### Increase Storage

To increase EBS volume size:

1. Edit `infrastructure/airbnb_pipeline/airbnb_stack.py`
2. Find `volume_size=30` and change to desired size
3. Deploy: `cdk deploy`

---

## 📊 Monitoring & Maintenance

### Daily Health Check

```bash
# Check EC2 instance status
aws ec2 describe-instance-status \
  --filters "Name=tag:Project,Values=Airbnb*" \
  --region us-east-1 \
  --query 'InstanceStatuses[0].[InstanceStatus.Status,SystemStatus.Status]'

# Expected: ok ok
```

### Weekly Cost Check

```bash
# Estimate costs (using AWS Pricing Calculator)
# t3.medium: $0.04/hour × 40 hours/week = $1.60/week
# S3 storage: ~$0.023 per GB-month (estimate $1–2/month for 50–100 GB)
# Data transfer: minimal (local S3 operations)
# Total: ~$3–5/week
```

### Monthly Cleanup

```bash
# Delete old Parquet files to save S3 costs
aws s3 ls s3://airbnb-market-intel-silver-123456789012 --recursive --summarize

# Delete files older than 30 days (optional)
# Or set S3 lifecycle policy to delete/archive old data
```

---

## 🔒 Security Considerations

### 1. IAM Role Permissions

The EC2 instance has **minimal scoped permissions**:
- ✅ Read/write to **only** Bronze and Silver buckets
- ✅ **Cannot** access other AWS services
- ✅ **Cannot** create IAM roles or modify policies

### 2. S3 Bucket Security

- ✅ Buckets are **private** (no public access)
- ✅ **Encryption at rest** (AES256)
- ✅ **Versioning enabled** (protect against accidental deletion)
- ✅ **SSL/TLS enforced** (in-transit encryption)

### 3. Network Security

- ✅ EC2 instance in **public subnet** (for internet access)
- ✅ **Restricted security group** (SSH only from your IP)
- ✅ Use **Systems Manager Session Manager** instead of SSH (no key pairs)
- ✅ **Outbound HTTPS/HTTP allowed** (for Inside Airbnb downloads)

### 4. Secrets Management

- ⚠️ **DO NOT** store secrets in user data scripts
- ✅ Use AWS Secrets Manager for Supabase credentials:

```bash
# Create secret
aws secretsmanager create-secret \
  --name airbnb/supabase \
  --secret-string '{
    "host": "your-project.supabase.co",
    "password": "your-password"
  }'

# Reference in CDK code (advanced)
```

---

## 🔧 Troubleshooting

### Issue: "cdk not found"

**Solution:**
```bash
npm install -g aws-cdk
cdk --version
```

### Issue: "Error: ENOENT: no such file or directory, open 'cdk.json'"

**Solution:**
```bash
cd infrastructure/
cdk synth
```

### Issue: "InvalidAction.NotAuthorized: You are not authorized to perform"

**Solution:** Check IAM permissions. Your user needs:
- `cloudformation:*`
- `ec2:*`
- `s3:*`
- `iam:*` (for role creation)

```bash
# Check current user permissions
aws iam get-user
```

### Issue: "An error occurred (AlreadyExistsException)"

**Solution:** S3 bucket names are globally unique. Add a random suffix:

```bash
# Edit cdk.json or airbnb_stack.py
# Change bucket name to: airbnb-market-intel-bronze-{{account}}-{{random}}
```

### Issue: "Deployment takes >10 minutes"

**Expected:** First deployment takes 5–10 minutes (CloudFormation stack creation).  
**Subsequent deployments:** 1–3 minutes (updates only).

---

## 📈 Cost Estimation

### Monthly Cost Breakdown

| Resource | Hourly | 40h/week | Monthly | Notes |
|----------|--------|----------|---------|-------|
| EC2 t3.medium | $0.0416 | $1.66 | $6.66 | Auto-stop after 8 PM |
| S3 Storage (Bronze) | - | - | $1.00 | ~50 GB, $0.023/GB |
| S3 Storage (Silver) | - | - | $1.00 | ~50 GB, $0.023/GB |
| S3 Requests | - | - | <$0.10 | Minimal API calls |
| Data Transfer | - | - | Free | EC2 to S3 is free |
| CloudWatch Logs | - | - | <$0.50 | 7-day retention |
| **Total** | - | - | **$8–10** | **Out of $100 budget** |

### Cost Optimization Tips

- ✅ Use **t3.small** ($0.0208/hour) instead of t3.medium to cut costs in half
- ✅ Enable **auto-stop schedule** (biggest savings)
- ✅ Use **S3 Intelligent-Tiering** for older data
- ✅ Enable **S3 Glacier transition** after 90 days

---

## 📚 Next Steps

1. **Deploy:** Follow steps 1–5 above
2. **Connect:** SSH/SSM into EC2 instance
3. **Run Pipeline:** Execute `python src/ingest.py` (data will flow to S3)
4. **Monitor:** Watch CloudWatch logs for execution status
5. **Scale:** Add more cities to `config/cities.yaml` as needed

---

## 🆘 Support

- **AWS CDK Docs:** https://docs.aws.amazon.com/cdk/
- **CloudFormation Troubleshooting:** https://docs.aws.amazon.com/AWSCloudFormation/
- **EC2 Systems Manager:** https://docs.aws.amazon.com/systems-manager/
- **S3 Security:** https://docs.aws.amazon.com/s3/security/

---

**Happy deploying! 🚀**
