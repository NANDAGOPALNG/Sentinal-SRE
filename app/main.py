from fastapi import FastAPI, HTTPException
import logging
import os

app = FastAPI(title="SentinalSRE-Sandbox")

LOG_FILE = "/var/log/sentinal/app.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("ProdApp")

@app.get("/")
def read_root():
    logger.info("Health check performed.")
    return {"status": "online"}

@app.get("/crash")
def trigger_crash():
    """Manually triggers the system incident for agents."""
    logger.error("CRITICAL: Database connection pool exhausted. ConnectionTimeout at line 42.")
    raise HTTPException(status_code=500, detail="Internal Server Error")
