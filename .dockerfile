FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /root/.streamlit && \
    echo "[server]\nheadless = true\nport = 8501\naddress = 0.0.0.0" > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py"]