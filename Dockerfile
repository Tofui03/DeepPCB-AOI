# numpy==2.5.4 requires Python >= 3.12 (the viewer PRD sketch used 3.11).
# opencv-python-headless 5.x bundles its own media libraries and does not link
# libGL or libglib, so this image installs no apt packages.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DETECTOR_BACKEND=yolov8
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
