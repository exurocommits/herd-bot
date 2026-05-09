FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn websockets numpy pandas requests

COPY . .

ENV PORT=8090
EXPOSE 8090

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8090/api/health || exit 1

CMD ["python3", "server.py"]
