import requests
import time
import os

# The endpoint for submitting the image processing request
SUBMIT_URL = 'http://localhost:8889/controlnet/'
# The endpoints for checking the status and getting the result
STATUS_URL = 'http://localhost:8889/jobs/status/{job_id}'
RESULT_URL = 'http://localhost:8889/jobs/result/{job_id}'

def submit_image_processing(image_path, mask_path, coordinates, prompt=None):  
    files = {
        'image': ('image.png', open(image_path, 'rb'), 'image/png'),
        'mask': ('mask.png', open(mask_path, 'rb'), 'image/png'),
        'prompt': (None, prompt),  # The first element is the filename, which is None in this case
    }
    data = {
    }
    response = requests.post(SUBMIT_URL, files=files) # , data=data)
    return response.json()

def check_status(job_id):
    return requests.get(STATUS_URL.format(job_id=job_id)).json()
    # check response.status_code

def get_result(job_id, save_path):
    response = requests.get(RESULT_URL.format(job_id=job_id))

    if response.status_code == 200:
        # Assuming the endpoint sends the file directly and there's no redirect
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Image successfully downloaded and saved to {save_path}")
    else:
        # Handle potential error (job not completed, result not available, etc.)
        print("Status code: ", response.status_code)
        print(response)
        # error_info = response.json()
        # print(f"Error: {error_info.get('error', 'Unknown error occurred')}")

# Example usage
if __name__ == '__main__':
    # Define the command-line argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Submit an image processing job to the API')
    parser.add_argument('image_path', type=str, help='Path to the RGB image file')
    parser.add_argument('mask_path', type=str, help='Path to the boolean mask file')
    # parser.add_argument('coordinates', type=str, help='JSON string of 3D coordinates and labels')
    parser.add_argument('--prompt', type=str, default='', help='Optional prompt text')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    res = submit_image_processing(
        args.image_path, 
        args.mask_path, 
        # args.coordinates, 
        args.prompt
    )
    print("submitted with response", res)
    job_id = res.get('job_id')
    print(f"Job submitted. ID: {job_id}")

    if job_id:
        # Wait and check the status until the job is completed
        while True:
            status_response = check_status(job_id)
            job_status = status_response.get('status')
            print(f"Job status: {status_response}")
            if job_status == 'SUCCESS':
                # Retrieve and save the result
                save_path = 'output_image.png'
                get_result(job_id, save_path)
                break
            elif job_status == 'FAILURE':
                print(f"Error: {status_response.get('error')}")
                break
            time.sleep(2)
