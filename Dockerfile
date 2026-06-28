# Use Python 3.11 slim — smaller image, faster deploy
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker layer caching)
# If requirements don't change, this layer is cached
# Only reinstalls packages when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY agent/ ./agent/
COPY main.py .
COPY data/ ./data/

# Port the app runs on inside container
EXPOSE 8000

# Health check — Docker checks this every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" \
  || exit 1

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]