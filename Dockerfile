FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libgl1 libglib2.0-0 git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirement file first for better caching
COPY requirements.txt .

# Install Python dependencies (force pre-built wheels)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.21.5 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose port if needed (replace 5000 if needed)
EXPOSE 5000

CMD ["python", "pipeline_demo.ipynb"]
