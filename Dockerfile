FROM python:3.9-slim-buster
WORKDIR /service
COPY requirements_dev.txt .
COPY . ./
RUN pip install -r requirements_dev.txt
ENTRYPOINT ["python3", "app.py"]
