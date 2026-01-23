# 1. Base Image: Linux with Python 3.10 installed
FROM python:3.10-slim

# 2. Seup Work Directory: Create a folder called 'app' inside the container
WORKDIR /app

#3. Python Settings (Not Secrets)
ENV PYTHONDONTWRYTEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#4. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 5. COPY CODE
# Copy everything from laptop folder to the /app folder in the container
COPY . .

# 6. The start command
# This runs the app on port 8000 inside the container
CMD ["gunicorn", "src:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]