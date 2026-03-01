FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    espeak \
    && rm -rf /var/lib/apt/lists/*

COPY app /app

RUN pip install --no-cache-dir \
    flask \
    opencv-python \
    ultralytics \
    pyttsx3 \
    numpy

EXPOSE 5000

CMD ["python", "main.py"]

