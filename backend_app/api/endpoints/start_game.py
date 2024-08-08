import cv2
import random
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
import backend_app.schemas as schemas
from backend_app.api.utils_pixabay import download_image

router = APIRouter()


def canny_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    edge_path = image_path.replace('input.jpg', 'output.jpg')
    cv2.imwrite(edge_path, edges)
    return edge_path


@router.post("/start_game", status_code=201)
async def start_game():
    # keywords = ['cat', 'dog', 'car', 'tree', 'house']
    keywords = ['cat', 'dog']
    selected_keyword = random.choice(keywords)

    file_path = await download_image(selected_keyword)
    if file_path:
        edge_path = canny_edge_detection(file_path)
        return {
            "sketch_url": edge_path,
            "original_url": file_path,
            "keywords": keywords
        }
    else:
        raise HTTPException(status_code=404, detail="Image not found")

