from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import shutil
import asyncio
from fastapi.responses import FileResponse
from celery_app import celery_app, handle_image_processing
from celery.result import AsyncResult

from fastapi import HTTPException, status
from fastapi.responses import FileResponse

app = FastAPI()


@app.post("/controlnet/")
async def controlnet(
    background_tasks: BackgroundTasks,
    image: UploadFile, # = File(...),
    mask: UploadFile, # = File(...),
    # coordinates: List[Coordinate],
    prompt: str,
    seed: int,
    num_images: int,
    resolution: int,
    num_steps: int
):
    # Save files temporarily and handle them
    image_filename = f"/tmp/temp_{uuid4()}_request_img.png"
    mask_filename = f"/tmp/temp_{uuid4()}_request_mask.png"
    print(f"saving to {image_filename} and {mask_filename}")
    with open(image_filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    with open(mask_filename, "wb") as buffer:
        shutil.copyfileobj(mask.file, buffer)

    # Create a job ID and start the background task

    task = handle_image_processing.delay(image_filename, mask_filename, prompt, seed=seed, num_images=num_images, resolution=resolution, num_steps=num_steps)
    print(f"created job {task.id}")
    return {"job_id": task.id}


@app.get("/jobs/status/{job_id}")
async def get_job_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    if result.status == "FAILURE":
        try:
            res = result.get()
        except Exception as e:
            print("Exception: ", e)
            return {"job_id": job_id, "status": result.status, "error": str(e)}
    res_state = {"job_id": job_id, "status": result.status}
    if result.status == "PROGRESS" or result.status == "SUCCESS":
        if result.info:
            print(result.info)
            res_state.update(result.info)
    
    return res_state


@app.get("/jobs/result/{job_id}/{image_id}")
async def get_result(job_id: str, image_id: int):
    result = AsyncResult(job_id, app=celery_app)
    
    if result.state == 'PENDING':
        # The job did not start yet
        raise HTTPException(status_code=status.HTTP_202_ACCEPTED, detail="Task pending, try again later.")

    elif result.state != 'SUCCESS':
        # Something went wrong in the processing
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Task failed with status: {result.state}")

    try:
        res = result.get()
        num_images = res["num_images"] # len(res["image_filenames"])
        assert 0 <= image_id < num_images, f"image_id must be between 0 and {num_images}"
        image_path = res["image_filenames"][image_id]
        return FileResponse(image_path, media_type='image/png', filename=image_path.split("/")[-1])
    
    except Exception as e:
        # Handle specific exceptions here if necessary
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889, ssl_keyfile="./key.pem", ssl_certfile="./cert.pem")


