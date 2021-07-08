FROM python:3.8-slim-buster

WORKDIR /app
RUN pip install flask transformers torch

COPY . .

EXPOSE 5000

CMD ["python3", "main.py"]
