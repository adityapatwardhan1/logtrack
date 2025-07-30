# FROM python:3.10.12-slim

# # Optional: install additional Linux packages you may need
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# # Copy your code
# COPY . /app

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Command to run your app
# CMD ["python", "data/app.py"]
