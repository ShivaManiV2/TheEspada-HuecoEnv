FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/

# We'll use a hack to ensure dependencies are installed even if they're not in requirements.txt
# but common for this project
RUN pip install --no-cache-dir -r requirements.txt || pip install fastapi uvicorn pydantic numpy httpx openai

# Copy application files
COPY env /app/env
COPY agents /app/agents
COPY tasks /app/tasks
COPY server /app/server
COPY inference.py /app/
COPY simulate.py /app/
COPY train.py /app/
COPY openenv.yaml /app/

# Create data directory for world memory and logs
RUN mkdir -p /app/data

# OpenEnv HTTP port requirements / HF spaces default
EXPOSE 7860
ENV PORT=7860

# Command to run the application using uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
