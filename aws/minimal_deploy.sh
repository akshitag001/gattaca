#!/usr/bin/env bash
# Minimal single-instance deploy. Paste this whole file into the instance's terminal
# (EC2 Instance Connect, in-browser — no SSH key needed) after cloning/uploading the repo.
#
# Cheap-path deploy: no ALB, no Auto Scaling, no API Gateway, no COOL AMI.
# One t4g.small (Graviton/arm64) instance, stock Ubuntu, app in a venv, systemd keeps it up.
# This still satisfies "a meaningful component on AWS" + "a working web endpoint" — it is
# just the fast, cheap version. aws/infra/ (CDK) remains available for the fuller
# ALB+ASG+COOL architecture if there is time later.
set -euo pipefail

APP=/opt/gattaca

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends python3-venv python3-pip libgl1 libglib2.0-0 git

sudo mkdir -p "$APP"
sudo chown "$USER":"$USER" "$APP"

# repo is expected to already be at ~/gattaca (scp'd or git-cloned); adjust SRC if different
SRC="${1:-$HOME/gattaca}"
rsync -a --exclude .venv --exclude .git --exclude 'datasets/*/' --exclude demo \
      --exclude '__pycache__' --exclude '*.mp4' "$SRC"/ "$APP"/

cd "$APP"
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt

python scripts/fetch_models.py || echo "hand-landmark models unavailable; classical detector will be used"

python - <<'EOF'
import cv2, platform, sys
print(f"cv2 {cv2.__version__} from {cv2.__file__}")
print(f"python {sys.version.split()[0]} on {platform.machine()}")
EOF

sudo tee /etc/systemd/system/gattaca.service > /dev/null <<EOF
[Unit]
Description=GATTACA exposure-scoring service
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP
ExecStart=$APP/.venv/bin/uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 2 --no-access-log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now gattaca.service
sleep 2
sudo systemctl is-active --quiet gattaca.service && echo "gattaca.service is running"
curl -s localhost:8000/health && echo
echo
echo "Public URL (after opening port 8000 in the security group):"
echo "  http://$(curl -s ifconfig.me):8000"
