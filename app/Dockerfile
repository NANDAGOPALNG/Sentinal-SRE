FROM python:3.12-slim
WORKDIR /app
RUN pip install fastapi uvicorn
COPY main.py .
RUN mkdir -p /var/log/sentinal
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
