# COOL benchmark: Cloud-Optimized OpenCV on Graviton vs stock OpenCV on x86

**Claim under test:** the GATTACA core workload (the OpenCV kernels it really calls,
end-to-end photo assessment, and the video agent) runs faster and cheaper on
COOL v3.1 (OpenCV 5.0) on Graviton4 than on stock `opencv-contrib-python-headless==5.0.0.93`
on a comparable x86 instance.

## Method (reproducible)

| | Candidate | Baseline |
|---|---|---|
| Instance | `c8g.xlarge` (Graviton4, 4 vCPU) | `c7i.xlarge` (Sapphire Rapids, 4 vCPU) |
| Image | "Cloud Optimized OpenCV For AWS Graviton4" AMI v3.1 (Ubuntu 24.04) | Ubuntu 24.04 (Canonical SSM parameter) |
| OpenCV | COOL `cv2` in `/opt/cool/venvs/python_3.12` | pip `opencv-contrib-python-headless==5.0.0.93` |
| Inputs | seeded synthetic photos and clips (`gattaca.synth`), identical on both | same |

Steps:

```bash
# 0. package the app and deploy (or launch the two instances by hand)
tar czf gattaca.tar.gz --exclude .venv --exclude 'datasets/*/' .
aws s3 cp gattaca.tar.gz s3://<artifact-bucket>/
cd aws/infra && cdk deploy -c coolAmiId=<ami> -c artifactBucket=<bucket> -c withX86Baseline=true

# 1. on the COOL instance (SSM session)
source /opt/cool/venvs/python_3.12/bin/activate
cd /opt/gattaca && python aws/cool_benchmark/bench.py --label cool-c8g.xlarge

# 2. on the x86 instance
cd /opt/gattaca && /opt/stock/bin/python aws/cool_benchmark/bench.py --label stock-c7i.xlarge

# 3. anywhere
python aws/cool_benchmark/compare.py results/cool-c8g.xlarge.json results/stock-c7i.xlarge.json \
       --price-a <c8g $/h> --sw-a <COOL $/h> --price-b <c7i $/h>
```

## What counts as evidence that COOL ran the workload

Every result JSON records `provenance`: `cv2.__file__` (must sit under `/opt/cool/...`),
`sys.prefix`, the KleidiCV/NEON lines of `cv2.getBuildInformation()`, `platform.machine()`
(`aarch64`) and the EC2 `instance-type` from IMDSv2. The service host also keeps
`cool_build_info.txt`, written at boot by `bootstrap_cool.sh`, and `/health` reports
the OpenCV version that serves traffic.

## Results

*Pending: run on AWS. Paste `compare_*.md` here.* A local run on the dev laptop
(`--label local-dev`) only checks that the harness works; it is **not** a COOL result.
