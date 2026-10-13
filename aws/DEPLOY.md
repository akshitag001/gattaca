# Deploying GATTACA to AWS

Running a meaningful component on AWS is an **eligibility requirement** for the competition, not a bonus. Without it the entry cannot win any prize, including 1st–3rd place. This runbook takes you from nothing to a live endpoint judges can hit.

Architecture: **API Gateway → ALB → Auto Scaling group of Graviton4 (c8g) instances running the COOL AMI**, with S3 for datasets, results and transient uploads. Diagram: [`../docs/architecture.md`](../docs/architecture.md).

Budget roughly **2–3 hours** for the first deploy.

---

## Phase 0 — Accounts and tools (~40 min)

### 0.1 AWS account

Create one at <https://aws.amazon.com/free/>. The competition notes new customers get **$100 in Free Tier credits**, plus up to $100 more through eligible activities. Claim them — this deployment is **not** Free-Tier-eligible (Graviton compute and the ALB both bill).

### 0.2 Install the AWS CLI

Not currently installed on this machine. Windows:

```powershell
winget install -e --id Amazon.AWSCLI
# or download the MSI: https://awscli.amazonaws.com/AWSCLIV2.msi
```

Reopen your terminal, then confirm:

```bash
aws --version          # expect: aws-cli/2.x
```

### 0.3 Credentials

In the AWS console: **IAM → Users → Create user → Attach `AdministratorAccess`** (acceptable for a short-lived competition account; scope it down for anything long-lived), then **Security credentials → Create access key → CLI**.

```bash
aws configure
#   AWS Access Key ID:     <paste>
#   AWS Secret Access Key: <paste>
#   Default region name:   us-east-1
#   Default output format: json

aws sts get-caller-identity      # must print your Account and Arn
```

Keep the region consistent everywhere below. `us-east-1` has the widest Graviton4 and Marketplace coverage.

### 0.4 Node.js

Already installed (v24). The CDK CLI runs via `npx`, so nothing else is needed.

---

## Phase 1 — Subscribe to COOL (~15 min)

The COOL AMI is a paid AWS Marketplace product with a **7-day free trial**. The subscription must be accepted *before* any instance can launch from it, or the Auto Scaling group fails with `OptInRequired`.

1. Open **Cloud Optimized OpenCV For AWS Graviton4** on AWS Marketplace:
   <https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs>
2. **Continue to Subscribe** → accept terms. Subscription can take a few minutes to activate.
3. **Continue to Configuration** → choose **your region** → copy the **AMI ID** (`ami-0...`).

```bash
export AWS_REGION=us-east-1
export COOL_AMI=ami-xxxxxxxxxxxxxxxxx     # from step 3
```

Verify the AMI is visible to your account:

```bash
aws ec2 describe-images --image-ids $COOL_AMI --region $AWS_REGION \
  --query 'Images[0].{Name:Name,Arch:Architecture,State:State}'
# expect Architecture = arm64, State = available
```

> **If COOL blocks you:** you can still deploy and stay eligible for the main prizes by using a stock ARM Ubuntu AMI (see [Fallback](#fallback-deploy-without-cool)). You would forfeit only the $1,000 COOL award. Do not let COOL hold up eligibility.

---

## Phase 2 — Upload the application (~10 min)

The instances download the app from S3 at boot, so **the tarball must exist before you deploy**.

```bash
cd /c/Users/24bcscs005/Downloads/GATTACA

# bucket names are globally unique — add something of your own
export ARTIFACT_BUCKET=gattaca-artifact-akshit-$(date +%s)
aws s3 mb s3://$ARTIFACT_BUCKET --region $AWS_REGION

tar --exclude='.venv' --exclude='.git' --exclude='aws/infra/.venv-cdk' \
    --exclude='aws/infra/cdk.out' --exclude='datasets/*/' --exclude='demo' \
    --exclude='__pycache__' --exclude='*.mp4' \
    -czf gattaca.tar.gz .

aws s3 cp gattaca.tar.gz s3://$ARTIFACT_BUCKET/gattaca.tar.gz
aws s3 ls s3://$ARTIFACT_BUCKET/          # confirm it landed
```

---

## Phase 3 — Deploy (~30 min, mostly waiting)

```bash
cd aws/infra
./.venv-cdk/Scripts/python.exe -m pip install -r requirements.txt   # already done once
export PATH="$PWD/.venv-cdk/Scripts:$PATH"

# one-time per account+region: creates the CDK staging bucket and roles
npx -y aws-cdk@2 bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/$AWS_REGION

npx -y aws-cdk@2 deploy \
  -c coolAmiId=$COOL_AMI \
  -c artifactBucket=$ARTIFACT_BUCKET \
  -c region=$AWS_REGION \
  --require-approval never
```

CDK prints the stack outputs on success. Save them:

```
ApiUrl              https://xxxxxxxx.execute-api.us-east-1.amazonaws.com   <- the judge-facing URL
AlbDns              Gattaca-Alb-....elb.amazonaws.com
UploadsBucket / DatasetsBucket / ResultsBucket / ArtifactBucketName
```

Instances take **5–10 minutes** after the stack completes to finish `bootstrap_cool.sh` and pass ALB health checks. Be patient before concluding it is broken.

---

## Phase 4 — Verify

```bash
export API=<ApiUrl from the outputs>
curl -s $API/health | python -m json.tool
```

Expect `{"status":"ok","opencv":"5.0.0", ...}`. Then open `$API` in a browser — the full landing page and the upload tool should work.

### If it does not come up

```bash
# find the instance
aws ec2 describe-instances --region $AWS_REGION \
  --filters "Name=tag:gattaca:role,Values=cool-graviton-core" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].InstanceId' --output text

# connect without SSH keys (the role already grants SSM)
aws ssm start-session --target <instance-id> --region $AWS_REGION
```

On the instance:

```bash
sudo systemctl status gattaca            # our service
sudo journalctl -u gattaca -n 50         # app logs
sudo tail -100 /var/log/cloud-init-output.log   # boot script output
cat /opt/gattaca/cool_build_info.txt     # proof of which OpenCV build is installed
```

Common causes: Marketplace subscription not active yet; the tarball missing from the artifact bucket; the health check needing more time.

---

## Phase 5 — The COOL benchmark (the $1,000 award)

The award requires evidence that **COOL executed the core workload on Arm**, plus reproducible measurements against a baseline.

### 5.1 On the Graviton instance

```bash
aws ssm start-session --target <instance-id> --region $AWS_REGION

sudo su - ubuntu
source /opt/cool/venvs/python_3.12/bin/activate
cd /opt/gattaca
python aws/cool_benchmark/bench.py --label cool-c8g.xlarge
aws s3 cp aws/cool_benchmark/results/cool-c8g.xlarge.json s3://$RESULTS_BUCKET/
```

The result JSON records `cv2.__file__` (must be under `/opt/cool/...`), `sys.prefix`, the KleidiCV build flags, `platform.machine()` = `aarch64`, and the EC2 instance type from IMDS. **That provenance block is your evidence.**

### 5.2 The x86 baseline

Redeploy with the baseline instance, then run the identical script:

```bash
npx -y aws-cdk@2 deploy -c coolAmiId=$COOL_AMI -c artifactBucket=$ARTIFACT_BUCKET \
  -c region=$AWS_REGION -c withX86Baseline=true --require-approval never
```

```bash
# on the x86 box
/opt/stock/bin/python aws/cool_benchmark/bench.py --label stock-c7i.xlarge
```

### 5.3 Compare

```bash
python aws/cool_benchmark/compare.py \
  results/cool-c8g.xlarge.json results/stock-c7i.xlarge.json \
  --price-a <c8g $/h> --sw-a <COOL $/h> --price-b <c7i $/h>
```

Verify current prices on the EC2 pricing page and the Marketplace listing — do not publish the placeholder defaults. Paste the generated table into [`cool_benchmark/README.md`](cool_benchmark/README.md) and the technical report.

---

## Cost control

Rough order of magnitude; **verify against current pricing** before relying on it:

| Item | Approx. |
|---|---|
| c8g.xlarge on-demand | ~$0.15/hr |
| COOL AMI software fee | see listing (c8g sizes are a few cents/hr) |
| Application Load Balancer | ~$0.023/hr + LCU |
| S3 / API Gateway at demo volume | cents |

Running 24/7 from deploy through judging (Oct 26 → Nov 9) is roughly **$70–100**, which the $100 Free Tier credits plus a compute grant can cover — but watch it.

To trim: use `-c instanceType=c8g.large`, and tear down the x86 baseline once its benchmark numbers are captured.

```bash
# set a billing alarm early
aws budgets create-budget --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{"BudgetName":"gattaca","BudgetLimit":{"Amount":"120","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}'
```

**Keep the stack running through judging** (to Nov 9) — the rules require a working endpoint or an arranged live demo. Afterwards:

```bash
cd aws/infra && npx -y aws-cdk@2 destroy
```

`DatasetsBucket` and `ResultsBucket` have `RemovalPolicy.RETAIN` and survive on purpose; empty and delete them by hand if you want the charges gone.

---

## Fallback: deploy without COOL

If the Marketplace subscription stalls, stay eligible by deploying on stock ARM Ubuntu. You lose only the COOL award.

```bash
COOL_AMI=$(aws ssm get-parameter --region $AWS_REGION \
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
  --query 'Parameter.Value' --output text)
```

Deploy exactly as in Phase 3 with that AMI. `bootstrap_cool.sh` detects the missing COOL venv and builds its own from `requirements.txt` (stock OpenCV wheel). The service still runs; `/opt/gattaca/opencv_mode.txt` will read `stock`. Say plainly in the report that COOL was not used.

---

## Checklist

- [ ] AWS CLI installed, `aws sts get-caller-identity` works
- [ ] COOL subscription active, AMI ID captured
- [ ] `gattaca.tar.gz` uploaded to the artifact bucket
- [ ] `cdk bootstrap` done for the account + region
- [ ] `cdk deploy` succeeded, outputs saved
- [ ] `$API/health` returns OpenCV 5.0.0
- [ ] Landing page and upload flow work from the public URL
- [ ] COOL benchmark run on Graviton, provenance captured
- [ ] x86 baseline run, `compare.py` table pasted into the report
- [ ] Billing alarm set
- [ ] **ApiUrl recorded in the Devpost submission**
