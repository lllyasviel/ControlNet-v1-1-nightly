import requests
import time
import os
import json
import sys

# The endpoint for submitting the image processing request
SUBMIT_URL = 'http://localhost:8889/controlnet/'
# The endpoints for checking the status and getting the result
STATUS_URL = 'http://localhost:8889/jobs/status/{job_id}'
RESULT_URL = 'http://localhost:8889/jobs/result/{job_id}/{img_id}'

def submit_image_processing(image_path, mask_path, prompt, resolution, num_images, seed, num_steps): 
    with open(image_path, 'rb') as r1, open(mask_path, 'rb') as r2:
        files = {
            'image': ('image.png', r1, 'image/png'),
            'mask': ('mask.png', r2, 'image/png'),
        }
        params = {
            'prompt': prompt,  
            'resolution': resolution,  
            'num_images': num_images,  
            'seed': seed,
            'num_steps': num_steps,
        }
        response = requests.post(SUBMIT_URL, params=params, files=files) # , data=data)
    return response # .json()

def check_status(job_id):
    return requests.get(STATUS_URL.format(job_id=job_id)).json()
    # check response.status_code

def get_result(job_id, save_path):
    status_obj = check_status(job_id)
    for i in range(status_obj["num_images"]):
        response = requests.get(RESULT_URL.format(job_id=job_id, img_id=i))
        print("Image", i)
        if response.status_code == 200:
            # Assuming the endpoint sends the file directly and there's no redirect
            p = save_path.format(img_id=i)
            with open(p, 'wb') as f:
                f.write(response.content)
            print(f"Image successfully downloaded and saved to {p}")
        else:
            # Handle potential error (job not completed, result not available, etc.)
            print("Status code: ", response.status_code)
            print("Error: ", response.json().get("detail"))

# Example usage
if __name__ == '__main__':
    # Define the command-line argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Submit an image processing job to the API')
    parser.add_argument('image_path', type=str, help='Path to the RGB image file')
    parser.add_argument('mask_path', type=str, help='Path to the boolean mask file')
    # parser.add_argument('coordinates', type=str, help='JSON string of 3D coordinates and labels')
    parser.add_argument('--prompt', type=str, default='', help='Optional prompt text')
    parser.add_argument('--job_id', type=str, help='Download resources only from a previous job')
    parser.add_argument('--resolution', type=int, default=1024, help='Image resolution (W/H)')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed to use')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of processing steps')
    # Parse the command-line arguments
    args = parser.parse_args()

    if args.job_id:
        job_id = args.job_id
    else:
        res = submit_image_processing(
            image_path=args.image_path,
            mask_path=args.mask_path,
            # args.coordinates,
            prompt=args.prompt,
            resolution=args.resolution,
            num_images=args.num_images,
            seed=args.seed,
            num_steps=args.num_steps,
        )
        j = res.json()
        print("submitted with response:", json.dumps(j, indent=4, sort_keys=True))
        if j.get("detail"):
            print("Error occurred")
            print(j)
            sys.exit(1)
        job_id = j.get('job_id')
        print(f"Job submitted. ID: {job_id}")

    if job_id:
        # Wait and check the status until the job is completed
        while True:
            status_response = check_status(job_id)
            job_status = status_response.get('status')
            print(f"Job status: {status_response}")
            if job_status == 'SUCCESS':
                # Retrieve and save the result
                save_path = 'output_image_{img_id}.png'
                get_result(job_id, save_path)
                break
            elif job_status == 'FAILURE':
                print(f"Error: {status_response.get('error')}")
                break
            time.sleep(2)
