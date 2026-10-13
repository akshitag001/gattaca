#!/usr/bin/env bash
# Installs GATTACA on an EC2 instance and runs it as a systemd service.
#
# Preferred path: the COOL AMI (Cloud-Optimized OpenCV) on Graviton, whose venv already
# provides cv2 (OpenCV 5 + Arm/KleidiCV optimisations). We must NOT pip-install opencv
# on top of it, or the stock wheel would shadow COOL and invalidate the benchmark.
#
# Fallback path: any other AMI (e.g. stock Ubuntu arm64/x86) -> build our own venv from
# requirements.txt, which pins the stock opencv wheel. The service still runs and the
# entry stays AWS-eligible; only the COOL award is forfeited. See ../DEPLOY.md.
set -euo pipefail

APP=/opt/gattaca
COOL_VENV=${COOL_VENV:-/opt/cool/venvs/python_3.12}

if [ -x "$COOL_VENV/bin/python" ]; then
    MODE=cool
    VENV="$COOL_VENV"
    REQS="$APP/requirements-cool.txt"      # deliberately excludes opencv and numpy
else
    MODE=stock
    echo "WARNING: $COOL_VENV not found. Falling back to a stock OpenCV venv."
    echo "WARNING: the Best Use of COOL award cannot be claimed from this deployment."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends python3-venv python3-pip libgl1 libglib2.0-0
    VENV=/opt/gattaca-venv
    python3 -m venv "$VENV"
    REQS="$APP/requirements.txt"           # pins the stock opencv wheel
fi

source "$VENV/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r "$REQS"

python - <<'EOF'
import cv2, platform, sys
print(f"cv2 {cv2.__version__} from {cv2.__file__}")
print(f"python {sys.version.split()[0]} at {sys.prefix} on {platform.machine()}")
EOF

python "$APP/scripts/fetch_models.py" || echo "hand-landmark models unavailable; classical detector will be used"

# record which OpenCV build actually serves traffic (evidence for the COOL award)
python -c "import cv2; print(cv2.getBuildInformation())" > "$APP/cool_build_info.txt"
echo "$MODE" > "$APP/opencv_mode.txt"

cat > /etc/systemd/system/gattaca.service <<EOF
[Unit]
Description=GATTACA exposure-scoring service (OpenCV mode: $MODE)
After=network-online.target

[Service]
EnvironmentFile=-/etc/gattaca.env
WorkingDirectory=$APP
ExecStart=$VENV/bin/uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers $(nproc) --no-access-log
Restart=always
RestartSec=5
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now gattaca.service
systemctl is-active --quiet gattaca.service && echo "gattaca.service running ($MODE)"
