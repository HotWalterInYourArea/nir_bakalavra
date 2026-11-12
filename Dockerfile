FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed (e.g., for building packages)
# RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The code will be mounted via volume, but this ensures the container has a default command
CMD ["tail", "-f", "/dev/null"]