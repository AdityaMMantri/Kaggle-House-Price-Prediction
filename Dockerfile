# Use lightweight Python image
FROM python:3.10-bullseye

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies required by ML libs
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip + install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy full project
COPY . .

# Expose Streamlit and Prometheus ports
EXPOSE 8501
EXPOSE 8000

# Start Streamlit App
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]