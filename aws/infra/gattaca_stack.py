"""GATTACA on AWS.

  Internet ─▶ API Gateway (HTTP API, /score, /protect, /health, /)
                 │ HTTP proxy
                 ▼
             ALB (public) ─▶ Auto Scaling group: Graviton4 c8g, COOL AMI (OpenCV 5 + KleidiCV)
                               uvicorn service.api:app inside /opt/cool/venvs/python_3.12
                               (detect, enhance, Gabor, ECC, fusion, agent: the core workload)
  S3: uploads (1-day expiry backstop; the service deletes in-process), datasets (private),
      results (study CSVs, COOL benchmarks), artifact (app tarball)
  Optional x86 baseline instance (c7i, stock opencv-python) for the COOL benchmark.
"""

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

APP_PORT = 8000


def _private_bucket(scope, cid, **kw) -> s3.Bucket:
    return s3.Bucket(scope, cid, block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                     encryption=s3.BucketEncryption.S3_MANAGED, enforce_ssl=True, **kw)


class GattacaStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kw) -> None:
        super().__init__(scope, cid, **kw)
        ctx = self.node.try_get_context
        cool_ami = ctx("coolAmiId")
        if not cool_ami:
            raise ValueError("pass -c coolAmiId=<Cloud Optimized OpenCV for Graviton4 AMI id>")
        instance_type = ctx("instanceType") or "c8g.xlarge"
        artifact_bucket_name = ctx("artifactBucket")

        # ---------------------------------------------------------------- storage
        uploads = _private_bucket(
            self, "Uploads", removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(1),   # S3 minimum granularity
                                              abort_incomplete_multipart_upload_after=Duration.days(1))])
        datasets = _private_bucket(self, "Datasets", versioned=True, removal_policy=RemovalPolicy.RETAIN)
        results = _private_bucket(self, "Results", versioned=True, removal_policy=RemovalPolicy.RETAIN)
        artifact = (s3.Bucket.from_bucket_name(self, "Artifact", artifact_bucket_name) if artifact_bucket_name
                    else _private_bucket(self, "ArtifactBucket", removal_policy=RemovalPolicy.DESTROY,
                                         auto_delete_objects=True))

        # ---------------------------------------------------------------- network
        vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=0,
                      subnet_configuration=[ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC)])
        alb_sg = ec2.SecurityGroup(self, "AlbSg", vpc=vpc, description="public HTTP to ALB")
        alb_sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(80))
        app_sg = ec2.SecurityGroup(self, "AppSg", vpc=vpc, description="ALB to app")
        app_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(APP_PORT))

        # ---------------------------------------------------------------- compute (COOL on Graviton4)
        role = iam.Role(self, "AppRole", assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
                        managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                            "AmazonSSMManagedInstanceCore")])
        datasets.grant_read(role)
        results.grant_read_write(role)
        uploads.grant_read_write(role)
        artifact.grant_read(role)

        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "set -euxo pipefail",
            "apt-get update -y && apt-get install -y awscli || snap install aws-cli --classic || true",
            f"aws s3 cp s3://{artifact.bucket_name}/gattaca.tar.gz /tmp/gattaca.tar.gz",
            "mkdir -p /opt/gattaca && tar -xzf /tmp/gattaca.tar.gz -C /opt/gattaca",
            "bash /opt/gattaca/aws/infra/bootstrap_cool.sh",
            f"echo 'RESULTS_BUCKET={results.bucket_name}' >> /etc/gattaca.env",
        )
        lt = ec2.LaunchTemplate(
            self, "CoolLaunchTemplate",
            machine_image=ec2.MachineImage.generic_linux({self.region: cool_ami}),
            instance_type=ec2.InstanceType(instance_type), role=role, security_group=app_sg,
            user_data=user_data, require_imdsv2=True,
            block_devices=[ec2.BlockDevice(device_name="/dev/sda1",
                                           volume=ec2.BlockDeviceVolume.ebs(30, encrypted=True))])
        asg = autoscaling.AutoScalingGroup(
            self, "CoolAsg", vpc=vpc, launch_template=lt, min_capacity=1, max_capacity=4,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            health_check=autoscaling.HealthCheck.elb(grace=Duration.minutes(10)))
        asg.scale_on_cpu_utilization("Cpu60", target_utilization_percent=60)
        Tags.of(asg).add("gattaca:role", "cool-graviton-core")

        alb = elbv2.ApplicationLoadBalancer(self, "Alb", vpc=vpc, internet_facing=True, security_group=alb_sg,
                                            idle_timeout=Duration.seconds(120))
        listener = alb.add_listener("Http", port=80, open=False)
        listener.add_targets("App", port=APP_PORT, protocol=elbv2.ApplicationProtocol.HTTP, targets=[asg],
                             health_check=elbv2.HealthCheck(path="/health", healthy_http_codes="200"),
                             deregistration_delay=Duration.seconds(30))

        # ---------------------------------------------------------------- public API
        api = apigw.HttpApi(self, "Api", description="GATTACA exposure scoring",
                            cors_preflight=apigw.CorsPreflightOptions(
                                allow_origins=["*"], allow_methods=[apigw.CorsHttpMethod.ANY],
                                allow_headers=["*"]))
        api.add_routes(path="/{proxy+}", methods=[apigw.HttpMethod.ANY],
                       integration=integrations.HttpUrlIntegration(
                           "Alb", f"http://{alb.load_balancer_dns_name}/{{proxy}}"))
        api.add_routes(path="/", methods=[apigw.HttpMethod.GET],
                       integration=integrations.HttpUrlIntegration("AlbRoot", f"http://{alb.load_balancer_dns_name}/"))

        # ---------------------------------------------------------------- optional x86 baseline
        if str(ctx("withX86Baseline")).lower() == "true":
            x86 = ec2.Instance(
                self, "X86Baseline", vpc=vpc, instance_type=ec2.InstanceType(ctx("x86InstanceType") or "c7i.xlarge"),
                machine_image=ec2.MachineImage.from_ssm_parameter(
                    "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"),
                role=role, vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
                require_imdsv2=True)
            x86.add_user_data(
                "apt-get update -y && apt-get install -y python3-venv awscli || snap install aws-cli --classic",
                f"aws s3 cp s3://{artifact.bucket_name}/gattaca.tar.gz /tmp/g.tgz && mkdir -p /opt/gattaca "
                "&& tar -xzf /tmp/g.tgz -C /opt/gattaca",
                "python3 -m venv /opt/stock && /opt/stock/bin/pip install -r /opt/gattaca/requirements.txt")
            Tags.of(x86).add("gattaca:role", "x86-stock-baseline")
            CfnOutput(self, "X86InstanceId", value=x86.instance_id)

        CfnOutput(self, "ApiUrl", value=api.api_endpoint, description="judge-facing endpoint (web UI at /)")
        CfnOutput(self, "AlbDns", value=alb.load_balancer_dns_name)
        CfnOutput(self, "UploadsBucket", value=uploads.bucket_name)
        CfnOutput(self, "DatasetsBucket", value=datasets.bucket_name)
        CfnOutput(self, "ResultsBucket", value=results.bucket_name)
        CfnOutput(self, "ArtifactBucketName", value=artifact.bucket_name)
