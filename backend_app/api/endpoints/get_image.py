
import random
from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter()


@router.get("/", status_code=200)
async def serve_image(file_path: str):
    return FileResponse(file_path)
