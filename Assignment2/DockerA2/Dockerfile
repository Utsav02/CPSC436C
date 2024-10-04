FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY image_classification.py /app/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "image_classification.py"]
