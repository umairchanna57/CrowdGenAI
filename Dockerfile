FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Install system dependencies and Git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with an increased timeout and a faster mirror
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Expose port 5000
EXPOSE 5000

# Command to run Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
