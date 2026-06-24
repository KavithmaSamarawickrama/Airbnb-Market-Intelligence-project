# AWS Resources Provisioned by CDK Stack

Complete list of AWS resources created when you run `cdk deploy`.

---

## 📊 Resource Inventory

### Compute (EC2)
- **1× EC2 t3.medium instance**
  - AMI: Amazon Linux 2 (latest)
  - Root volume: 30 GB GP3
  - Auto-stop at 8 PM UTC, auto-start 8 AM UTC (weekdays)
  - Security group: Restrictive ingress (SSH only), outbound HTTPS/HTTP

### Storage (S3)
- **1× S3 Bronze bucket** (`airbnb-market-intel-bronze-{account}`)
  - Versioning: Enabled
  - Encryption: AES256
  - Public access: Blocked
  - Lifecycle: Transition to Glacier after 90 days
  - Retention: Indefinite (for replayability)

- **1× S3 Silver bucket** (`airbnb-market-intel-silver-{account}`)
  - Versioning: Enabled
  - Encryption: AES256
  - Public access: Blocked
  - Lifecycle: Transition to Standard-IA after 30 days
  - Retention: Indefinite

### Networking (VPC)
- **1× VPC** (`airbnb-market-intel-vpc`)
  - CIDR: 10.0.0.0/16
  - 1 Availability Zone (cost optimization)

- **1× Public Subnet**
  - CIDR: 10.0.0.0/24
  - Route to Internet Gateway

- **1× Internet Gateway**
  - Attached to VPC
  - Routes public subnet traffic to internet

- **1× Route Table**
  - Public routes (0.0.0.0/0 → IGW)

### Security
- **1× Security Group** (`airbnb-market-intel-sg`)
  - Ingress: SSH (port 22) from 0.0.0.0/0 (restrict in production)
  - Egress: HTTPS (port 443) to 0.0.0.0/0
  - Egress: HTTP (port 80) to 0.0.0.0/0

- **1× IAM Role** (`airbnb-market-intel-ec2-role`)
  - Trust: EC2 service principal
  - Permissions:
    - s3:GetObject, PutObject, ListBucket, DeleteObject (Bronze/Silver only)
    - logs:CreateLogStream, PutLogEvents (/airbnb/* only)
    - ssm:* (Systems Manager Session Manager)

- **1× EventBridge IAM Role** (`airbnb-market-intel-eventbridge-role`)
  - Trust: EventBridge service principal
  - Permissions:
    - ec2:StartInstances, StopInstances (instance with matching tags)

### Monitoring (CloudWatch)
- **1× Log Group** (`/airbnb/pipeline/execution`)
  - Retention: 7 days
  - Removal policy: Destroy on stack deletion

### Orchestration (EventBridge)
- **1× EventBridge Rule** (`airbnb-market-intel-auto-stop-rule`)
  - Schedule: 8:00 PM UTC daily (20:00 UTC)
  - Action: EC2 StopInstances
  - Target: EC2 instance with Airbnb tag

- **1× EventBridge Rule** (`airbnb-market-intel-auto-start-rule`)
  - Schedule: 8:00 AM UTC weekdays only (08:00 UTC Mon-Fri)
  - Action: EC2 StartInstances
  - Target: EC2 instance with Airbnb tag

---

## 🏷️ Tags Applied

All resources are tagged with:

```json
{
  "Project": "Airbnb Market Intelligence",
  "Environment": "Development",
  "ManagedBy": "CDK",
  "CostCenter": "DataEngineering"
}
```

Use these tags for:
- Cost allocation
- Filtering in AWS Console
- Automation (e.g., EventBridge rules)
- Compliance tracking

---

## 📈 Resource Count Summary

| Category | Count | Notes |
|----------|-------|-------|
| Compute | 1 | EC2 t3.medium |
| Storage | 2 | S3 Bronze + Silver |
| Networking | 4 | VPC, Subnet, IGW, Route Table |
| Security | 3 | SG, IAM Role (EC2), IAM Role (EventBridge) |
| Monitoring | 1 | CloudWatch Log Group |
| Orchestration | 2 | EventBridge Rules |
| **Total Resources** | **13** | - |

---

## 💾 Storage Capacity

| Resource | Size | Notes |
|----------|------|-------|
| EC2 Root Volume | 30 GB | Sufficient for OS + Python + small datasets |
| S3 Bronze | Unlimited | (You pay for storage) |
| S3 Silver | Unlimited | (You pay for storage) |
| Local DuckDB | ~5 GB | On EC2 instance (/opt/airbnb-pipeline) |
| CloudWatch Logs | ~500 MB | 7-day retention, auto-purge |

**Total Monthly Storage (est.):** 100–200 GB = $2–5/month

---

## 🔍 How to View Resources

### AWS Console
```
https://console.aws.amazon.com/
→ EC2 → Instances (find airbnb-market-intel-instance)
→ S3 → Buckets (find airbnb-market-intel-bronze-*, silver-*)
→ CloudWatch → Logs (find /airbnb/pipeline/execution)
→ EventBridge → Rules (find airbnb-market-intel-auto-*)
```

### AWS CLI
```bash
# List EC2 instances
aws ec2 describe-instances --filters "Name=tag:Project,Values=Airbnb*"

# List S3 buckets
aws s3 ls | grep airbnb-market-intel

# List IAM roles
aws iam list-roles --query 'Roles[?contains(RoleName, `airbnb`)]'

# List security groups
aws ec2 describe-security-groups --filters "Name=tag:Project,Values=Airbnb*"

# List EventBridge rules
aws events list-rules --name-prefix airbnb-market-intel

# List CloudWatch log groups
aws logs describe-log-groups --log-group-name-prefix /airbnb
```

### CloudFormation
```bash
# View stack resources
aws cloudformation describe-stack-resources --stack-name AirbnbPipelineStack

# Get stack outputs
aws cloudformation describe-stacks --stack-name AirbnbPipelineStack \
  --query 'Stacks[0].Outputs'
```

---

## ⚠️ Important Notes

### Billing
- **EC2**: Charged per hour (on or off)
- **S3 Storage**: Charged per GB-month (versioning adds 2x)
- **S3 Requests**: Charged per 1,000 requests
- **Data Transfer**: FREE (EC2↔S3 in same region)
- **EventBridge**: $0.10 per rule per month

### Deletion
To delete all resources:
```bash
cdk destroy
```

**Note:** S3 buckets with data will block deletion. Empty first:
```bash
aws s3 rm s3://airbnb-market-intel-bronze-{account} --recursive
aws s3 rm s3://airbnb-market-intel-silver-{account} --recursive
cdk destroy
```

### Quotas & Limits
- EC2 default limit: 20 instances (no issue here)
- S3 buckets per account: 100 (no issue here)
- IAM roles per account: 1,000 (no issue here)

All limits are sufficient for this project.

---

## 📋 Pre-Deployment Checklist

Before deploying, verify:

- [ ] AWS account created
- [ ] AWS CLI configured (`aws sts get-caller-identity` works)
- [ ] Node.js installed (`npm --version` works)
- [ ] AWS CDK CLI installed (`cdk --version` works)
- [ ] Python 3.11+ installed
- [ ] IAM user has permissions (cloudformation, ec2, s3, iam, logs, events)
- [ ] AWS account has available quota for resources

---

## 📞 Support

If you need to:
- **Modify resources:** Edit `infrastructure/airbnb_pipeline/airbnb_stack.py` and `cdk deploy`
- **Delete resources:** Run `cdk destroy`
- **Scale resources:** Change instance type in `infrastructure/cdk.json`
- **Troubleshoot:** Check AWS CloudFormation events or CloudWatch Logs

See [infrastructure/CDK_DEPLOYMENT_GUIDE.md](infrastructure/CDK_DEPLOYMENT_GUIDE.md) for detailed instructions.

---

Generated: June 20, 2026  
Version: 1.0.0
