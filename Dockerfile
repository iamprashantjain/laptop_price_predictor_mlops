# # base image
# FROM python:3.9

# # working dir
# WORKDIR /app

# # copy
# COPY . /app

# # run
# RUN pip install -r requirements.txt

# # ports
# EXPOSE 5000

# # command to execute
# # CMD ["python","./flask_app/app.py"]
# #add gunicorn to handle multiple requests at a point

# CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]


# ===========


# Use slim version for smaller image size
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Run Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]