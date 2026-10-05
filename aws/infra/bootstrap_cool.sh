#!/usr/bin/env bash
# Installs GATTACA into the COOL (Cloud-Optimized OpenCV) Python environment on a
# Graviton instance launched from the COOL AMI, and runs it as a systemd service.
# The COOL venv already provides cv2 (OpenCV 5 + Arm optimisations): we must NOT
# pip-install opencv on top of it, or the stock wheel would shadow COOL.
set -euo pipefail

APP=/opt/gattaca
COOL_VENV=${COOL_VENV:-/opt/cool/venvs/python_3.12}

source "$COOL_VENV/bin/activate"
python - <<'EOF'
import cv2, sys
print("COOL cv2", cv2.__version__, "from", cv2.__file__, "python", sys.version.split()[0])
EOF
pip install --no-cache-dir -r "$APP/requirements-cool.txt"
python "$APP/scripts/fetch_models.py" || echo "hand-landmark models unavailable; classical detector will be used"

# prove which OpenCV build serves traffic (kept for the COOL-award evidence pack)
python -c "import cv2; print(cv2.getBuildInformation())" > "$APP/cool_build_info.txt"

cat > /etc/systemd/system/gattaca.service <<EOF
[Unit]
Description=GATTACA exposure-scoring service (COOL on Graviton)
After=network-online.target

[Service]
EnvironmentFile=-/etc/gattaca.env
WorkingDirectory=$APP
ExecStart=$COOL_VENV/bin/uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers $(nproc) --no-access-log
Restart=always
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now gattaca.service
