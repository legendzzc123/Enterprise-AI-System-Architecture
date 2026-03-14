import time
import uuid
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import redis

# Structured Logging Configuration
logging.basicConfig(level=logging.INFO, format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "request_id": "%(name)s", "message": "%(message)s"}')
logger = logging.getLogger("inference_service")

# Prometheus Metrics
INFERENCE_REQUEST_COUNT = Counter('inference_request_count', 'Total number of inference requests', ['endpoint'])
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Latency of model predictions', ['endpoint'])
CACHE_HITS = Counter('model_cache_hits', 'Total number of cache hits')

app = FastAPI(title="Enterprise AI Model Server", version="1.0.0")

# Redis Caching (Optional, configured via env)
try:
    cache = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
    logger.info("Connected to Redis for caching.")
except Exception as e:
    cache = None
    logger.warning(f"Redis not available: {e}")

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Request {request_id} processed in {duration:.4f}s")
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
@INFERENCE_LATENCY.labels(endpoint='/predict').time()
async def predict(data: Dict[str, Any]):
    INFERENCE_REQUEST_COUNT.labels(endpoint='/predict').inc()
    
    # Simple logic for enterprise-grade demo
    input_id = data.get("id")
    if not input_id:
        raise HTTPException(status_code=400, detail="Missing 'id' in request body")

    # Check Cache
    if cache:
        cached_result = cache.get(f"predict:{input_id}")
        if cached_result:
            CACHE_HITS.inc()
            return {"id": input_id, "prediction": cached_result, "source": "cache"}

    # Mock Model Inference
    time.sleep(0.05)  # Simulating high-performance inference latency
    prediction = {"score": 0.95, "class": "positive"}
    
    # Store in Cache
    if cache:
        cache.setex(f"predict:{input_id}", 3600, str(prediction))

    return {"id": input_id, "prediction": prediction, "source": "model"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
