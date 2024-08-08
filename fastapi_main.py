import logging

import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend_app.api.router import api_router
from backend_app.core.config import settings
from contextlib import asynccontextmanager

from cldm.ddim_hacked import DDIMSampler
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from global_model import ml_models

# logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Loading ML models...")
    ml_models["apply_canny"] = CannyDetector()

    model = create_model(settings.PATH_BASE_MODEL_V10).cpu()
    model.load_state_dict(load_state_dict(settings.WEIGHT_PATH_BASE_MODEL_V10, location='cuda'))
    model = model.cuda()

    ml_models["BASE_MODEL_V10"] = model
    ml_models["ddim_sampler"] = DDIMSampler(model)

    logger.info("Models loaded successfully.")
    yield
    # Clean up the ML models and release the resources
    logger.info("Cleaning up models...")
    # Clean up the ML models and release the resources
    ml_models.clear()

    logger.info("Models cleaned up.")

# app = FastAPI(openapi_url=f'{settings.PREFIX_URL}/openapi.json', docs_url=f'{settings.PREFIX_URL}/docs')

app = FastAPI(lifespan=lifespan,openapi_url=f'{settings.PREFIX_URL}/openapi.json', docs_url=f'{settings.PREFIX_URL}/docs')

origins = [
    "http://localhost:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# async def startup_event():
#     os.makedirs('./result_images', exist_ok=True)
#     os.makedirs('./result_images', exist_ok=True)

@app.get(f"{settings.PREFIX_URL}/")
async def root():
    return {"message": "This is ControlNet server!!"}


def start():
    uvicorn.run("fastapi_main:app", host="0.0.0.0", port=8000, reload=settings.SERVER_RELOAD)

app.include_router(api_router, prefix=settings.PREFIX_URL)

if __name__ == "__main__":
    start()
