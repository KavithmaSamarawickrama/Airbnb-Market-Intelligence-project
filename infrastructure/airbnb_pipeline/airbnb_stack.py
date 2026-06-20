"""
Core CDK Stack Definition for Airbnb Market Intelligence Pipeline

Provisions:
- VPC with public subnet
- EC2 t3.medium instance
- S3 Bronze & Silver buckets
- IAM role with minimal permissions
- Security groups
- CloudWatch logs
- Auto-stop schedule (cost optimization)
"""

import json
from typing import Any, Dict
import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_s3 as s3,
    aws_iam as iam,
    aws_logs as logs,
    aws_events as events,
    aws_events_targets as targets,
    aws_lambda as lambda_,
    core,
)
from constructs import Construct


class AirbnbPipelineStack(cdk.Stack):
    """Main CDK Stack for the Airbnb pipeline infrastructure."""

    def __init__(self, scope: Construct, id: str, config: Dict[str, Any], **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        self.config = config
        self.project_name = config["project_name"]
        self.environment = config["environment"]

        # Create infrastructure components
        self.vpc = self._create_vpc()
        self.security_group = self._create_security_group()
        self.iam_role = self._create_iam_role()
        self.s3_buckets = self._create_s3_buckets()
        self.ec2_instance = self._create_ec2_instance()
        self.log_group = self._create_cloudwatch_logs()
        self._setup_auto_stop_schedule()

    def _create_vpc(self) -> ec2.Vpc:
        """Create VPC with public subnet for EC2 instance."""
        vpc = ec2.Vpc(
            self,
            f"{self.project_name}-vpc",
            max_azs=1,  # Single AZ for cost optimization
            nat_gateways=0,  # No NAT gateway (cost optimization)
            cidr="10.0.0.0/16",
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PUBLIC,
                    name="public",
                    cidr_mask=24,
                )
            ],
        )
        return vpc

    def _create_security_group(self) -> ec2.SecurityGroup:
        """Create security group allowing SSH and HTTPS."""
        sg = ec2.SecurityGroup(
            self,
            f"{self.project_name}-sg",
            vpc=self.vpc,
            description="Security group for Airbnb pipeline EC2 instance",
            allow_all_outbound=True,
        )

        # Allow SSH from your IP (update below to restrict)
        # In production, use a bastion host or Systems Manager Session Manager
        sg.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(22),
            description="SSH access (restrict to your IP in production)",
        )

        # Allow HTTPS outbound (for Inside Airbnb downloads)
        sg.add_egress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(443),
            description="HTTPS outbound",
        )

        # Allow HTTP outbound (for Inside Airbnb downloads)
        sg.add_egress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(80),
            description="HTTP outbound",
        )

        return sg

    def _create_iam_role(self) -> iam.Role:
        """Create IAM role with minimal scoped permissions."""
        role = iam.Role(
            self,
            f"{self.project_name}-ec2-role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Role for Airbnb pipeline EC2 instance",
        )

        # S3 permissions (Bronze and Silver buckets only)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:DeleteObject",
                ],
                resources=[
                    f"arn:aws:s3:::{self.project_name}-bronze-*",
                    f"arn:aws:s3:::{self.project_name}-bronze-*/*",
                    f"arn:aws:s3:::{self.project_name}-silver-*",
                    f"arn:aws:s3:::{self.project_name}-silver-*/*",
                ],
                effect=iam.Effect.ALLOW,
            )
        )

        # S3 bucket creation (for initial setup)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:CreateBucket"],
                resources=["arn:aws:s3:::*"],
                conditions={
                    "StringLike": {
                        "s3:x-amz-bucket-region": [self.region],
                    }
                },
                effect=iam.Effect.ALLOW,
            )
        )

        # CloudWatch Logs permissions
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[f"arn:aws:logs:{self.region}:{self.account}:log-group:/airbnb/*"],
                effect=iam.Effect.ALLOW,
            )
        )

        # EC2 Systems Manager Session Manager (for secure shell access without SSH keys)
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")
        )

        return role

    def _create_s3_buckets(self) -> Dict[str, s3.Bucket]:
        """Create S3 buckets for Bronze and Silver layers."""
        buckets = {}

        # Bronze bucket (immutable, versioning enabled)
        bronze_bucket = s3.Bucket(
            self,
            f"{self.project_name}-bronze",
            bucket_name=f"{self.project_name}-bronze-{self.account}",
            versioning_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=cdk.Duration.days(90),
                        )
                    ]
                )
            ],
        )
        buckets["bronze"] = bronze_bucket

        # Silver bucket (processed data)
        silver_bucket = s3.Bucket(
            self,
            f"{self.project_name}-silver",
            bucket_name=f"{self.project_name}-silver-{self.account}",
            versioning_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.STANDARD_IA,
                            transition_after=cdk.Duration.days(30),
                        )
                    ]
                )
            ],
        )
        buckets["silver"] = silver_bucket

        return buckets

    def _create_ec2_instance(self) -> ec2.Instance:
        """Create EC2 t3.medium instance with Python and pipeline code."""
        # Use Ubuntu 22.04 LTS AMI
        ami = ec2.AmazonLinuxImage(
            generation=ec2.AmazonLinuxGeneration.AMAZON_LINUX_2,
            virtualization=ec2.AmazonLinuxVirt.HVM,
        )

        # User data script to bootstrap the instance
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "#!/bin/bash",
            "set -e",
            "echo 'Starting Airbnb Pipeline EC2 bootstrap...'",
            "",
            "# Update system",
            "yum update -y",
            "yum install -y python3.11 python3.11-devel git curl wget",
            "",
            "# Install Python dependencies",
            "python3.11 -m pip install --upgrade pip setuptools wheel",
            "",
            "# Clone repository (update URL as needed)",
            "cd /opt",
            "git clone https://github.com/KavithmaSamarawickrama/Airbnb-Market-Intelligence-project.git airbnb-pipeline",
            "cd airbnb-pipeline",
            "",
            "# Create Python venv",
            "python3.11 -m venv venv",
            "source venv/bin/activate",
            "",
            "# Install requirements",
            "pip install -r requirements.txt",
            "",
            "# Create systemd service for pipeline execution",
            "cat > /etc/systemd/system/airbnb-pipeline.service << 'EOF'",
            "[Unit]",
            "Description=Airbnb Market Intelligence Pipeline",
            "After=network.target",
            "",
            "[Service]",
            "Type=oneshot",
            "User=ec2-user",
            "WorkingDirectory=/opt/airbnb-pipeline",
            "Environment='PATH=/opt/airbnb-pipeline/venv/bin'",
            "ExecStart=/opt/airbnb-pipeline/venv/bin/python /opt/airbnb-pipeline/src/ingest.py",
            "StandardOutput=journal",
            "StandardError=journal",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "EOF",
            "",
            "# Create cron job for daily ingestion at 2 AM UTC",
            "echo '0 2 * * * /opt/airbnb-pipeline/venv/bin/python /opt/airbnb-pipeline/src/ingest.py >> /var/log/airbnb-pipeline.log 2>&1' | crontab -",
            "",
            "# Create CloudWatch Logs agent config",
            "mkdir -p /opt/aws/amazon-cloudwatch-agent/etc",
            "cat > /opt/aws/amazon-cloudwatch-agent/etc/config.json << 'EOF'",
            "{",
            '  "logs": {',
            '    "logs_collected": {',
            '      "files": {',
            '        "collect_list": [',
            '          {',
            '            "file_path": "/var/log/airbnb-pipeline.log",',
            '            "log_group_name": "/airbnb/pipeline/execution",',
            '            "log_stream_name": "{instance_id}"',
            '          }',
            '        ]',
            '      }',
            '    }',
            '  }',
            "}",
            "EOF",
            "",
            "echo 'EC2 bootstrap complete.'",
        )

        instance = ec2.Instance(
            self,
            f"{self.project_name}-instance",
            instance_type=ec2.InstanceType(self.config["instance_type"]),
            machine_image=ami,
            vpc=self.vpc,
            security_group=self.security_group,
            role=self.iam_role,
            key_name=None,  # Use Systems Manager Session Manager instead of SSH keys
            user_data=user_data,
            associate_public_ip_address=True,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=30,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        delete_on_termination=True,
                    ),
                )
            ],
        )

        return instance

    def _create_cloudwatch_logs(self) -> logs.LogGroup:
        """Create CloudWatch Log Group for pipeline execution logs."""
        log_group = logs.LogGroup(
            self,
            f"{self.project_name}-logs",
            log_group_name="/airbnb/pipeline/execution",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        return log_group

    def _setup_auto_stop_schedule(self) -> None:
        """Set up EventBridge rule to auto-stop EC2 instance on schedule (cost optimization)."""
        # Stop at 8 PM UTC daily
        stop_rule = events.Rule(
            self,
            f"{self.project_name}-auto-stop-rule",
            schedule=events.Schedule.cron(hour="20", minute="0", month="*", week_day="*", year="*"),
            description="Auto-stop Airbnb pipeline EC2 instance at 8 PM UTC",
        )

        stop_rule.add_target(
            targets.AwsApi(
                service="EC2",
                action="StopInstances",
                parameters={
                    "InstanceIds": [self.ec2_instance.instance_id],
                },
                role=self._create_eventbridge_role(),
            )
        )

        # Start at 8 AM UTC daily
        start_rule = events.Rule(
            self,
            f"{self.project_name}-auto-start-rule",
            schedule=events.Schedule.cron(hour="8", minute="0", month="*", week_day="1-5", year="*"),  # Weekdays only
            description="Auto-start Airbnb pipeline EC2 instance at 8 AM UTC (weekdays)",
        )

        start_rule.add_target(
            targets.AwsApi(
                service="EC2",
                action="StartInstances",
                parameters={
                    "InstanceIds": [self.ec2_instance.instance_id],
                },
                role=self._create_eventbridge_role(),
            )
        )

    def _create_eventbridge_role(self) -> iam.Role:
        """Create IAM role for EventBridge to control EC2 instance."""
        role = iam.Role(
            self,
            f"{self.project_name}-eventbridge-role",
            assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
            description="Role for EventBridge to start/stop EC2 instance",
        )

        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ec2:StartInstances",
                    "ec2:StopInstances",
                ],
                resources=[
                    f"arn:aws:ec2:{self.region}:{self.account}:instance/*",
                ],
                conditions={
                    "StringEquals": {
                        "aws:ResourceTag/Project": ["Airbnb Market Intelligence"],
                    }
                },
                effect=iam.Effect.ALLOW,
            )
        )

        return role

    def _create_eventbridge_role(self) -> iam.Role:
        """Create or reuse IAM role for EventBridge to control EC2 instance."""
        if not hasattr(self, "_eventbridge_role"):
            self._eventbridge_role = iam.Role(
                self,
                f"{self.project_name}-eventbridge-role",
                assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
                description="Role for EventBridge to start/stop EC2 instance",
            )

            self._eventbridge_role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "ec2:StartInstances",
                        "ec2:StopInstances",
                    ],
                    resources=[
                        f"arn:aws:ec2:{self.region}:{self.account}:instance/*",
                    ],
                    effect=iam.Effect.ALLOW,
                )
            )

        return self._eventbridge_role
