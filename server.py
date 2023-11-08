from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import shutil
import asyncio
from fastapi.responses import FileResponse
from celery_app import celery_app, handle_image_processing
from celery.result import AsyncResult

app = FastAPI()


@app.post("/controlnet/")
async def controlnet(background_tasks: BackgroundTasks,
                        image: UploadFile, # = File(...),
                        mask: UploadFile, # = File(...),
                        # coordinates: List[Coordinate],
                        # prompt: str
                    ):
    prompt = "red drawer"
    print("Using prompt: ", prompt)
    # Save files temporarily and handle them
    image_filename = f"/tmp/temp_{uuid4()}_request_img.png"
    mask_filename = f"/tmp/temp_{uuid4()}_request_mask.png"
    print(f"saving to {image_filename} and {mask_filename}")
    with open(image_filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    with open(mask_filename, "wb") as buffer:
        shutil.copyfileobj(mask.file, buffer)

    # Create a job ID and start the background task

    task = handle_image_processing.delay(image_filename, mask_filename, prompt)
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
    if result.status == "PROGRESS":
        if result.info:
            print(result.info)
            res_state.update(result.info)
    return res_state

@app.get("/jobs/result/{job_id}")
async def get_result(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    try:
        res = result.get()
    except Exception as e:
        print("Exception: ", e)
        return {"job_id": job_id, "status": result.status, "error": str(e)}

    if result.status != "SUCCESS":
        return {"job_id": job_id, "status": result.status, "error": "Result not available or job not completed."}

    image_path = res["image_filename"]
    return FileResponse(image_path, media_type='image/png', filename=image_path.split("/")[-1])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)


