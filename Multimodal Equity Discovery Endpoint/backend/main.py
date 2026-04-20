from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from db.database import engine, Base
from routes import instruments
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")
    yield

app = FastAPI(
    title="AAAI Multimodal Equity Discovery Endpoint",
    description="Deterministic retrieval and arithmetic assembly of closing price, time index, and activity frequency streams per instrument.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(instruments.router, prefix="/api/v1", tags=["instruments"])


@app.get("/health")
def health():
    return {"status": "ok", "module": "MultimodalEquityDiscovery"}
