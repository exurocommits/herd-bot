FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn websockets numpy pandas requests

COPY . .

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -f http://localhost:8090/api/health || exit 1

CMD ["python3", "server.py"]
