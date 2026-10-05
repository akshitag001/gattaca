# GATTACA service image: stock OpenCV 5 wheel, multi-arch (linux/amd64 + linux/arm64).
#
#   docker buildx build --platform linux/amd64,linux/arm64 -t gattaca:0.1.0 .
#   docker run -p 8000:8000 gattaca:0.1.0
#
# x86 <-> Arm notes:
# * The pinned wheels (opencv-contrib-python-headless, numpy, scipy) publish manylinux
#   aarch64 builds, so this image runs unchanged on Graviton (ECS/Fargate arm64) and x86.
# * COOL (Cloud-Optimized OpenCV) ships as a Graviton AMI, not a wheel. Production runs
#   the COOL AMI directly (aws/infra/bootstrap_cool.sh). This container is the portable
#   baseline and the local/dev path, and is what the x86 benchmark compares against.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY gattaca gattaca
COPY service service
COPY datasets/__init__.py datasets/loader.py datasets/
COPY study study
COPY scripts scripts
COPY models models

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
