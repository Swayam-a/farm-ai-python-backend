import os
import uuid
import subprocess
import shutil
import traceback
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import uvicorn

# --- 1. Configuration and Initialization ---
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SERVER_API_KEY = os.getenv("SERVER_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, SERVER_API_KEY]):
    raise ValueError("FATAL ERROR: Missing required environment variables. Check your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI(title="Vegetation Health Mapping API")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# --- 2. API Key Authentication ---
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == SERVER_API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate API credentials.")

# --- 3. Request Data Model ---
class ImageProcessRequest(BaseModel):
    rgb_image_path: str
    nir_image_path: str


# --- 4. Local Testing Endpoint ---
@app.post("/process-local-images/")
def process_local_images():
    print("--- Starting Local Processing Job ---")
    project_path = os.getcwd()
    local_data_dir = os.path.join(project_path, "local_test_data")
    output_dir = os.path.join(project_path, "local_outputs")
    os.makedirs(output_dir, exist_ok=True)

    local_rgb_path = os.path.abspath(os.path.join(local_data_dir, "test_rgb.jpg"))
    local_nir_path = os.path.abspath(os.path.join(local_data_dir, "test_nir.jpg"))
    
    if not os.path.exists(local_rgb_path) or not os.path.exists(local_nir_path):
        raise HTTPException(
            status_code=404, 
            detail="Local test images not found. Place 'test_rgb.jpg' and 'test_nir.jpg' in the 'local_test_data' folder."
        )

    job_id = str(uuid.uuid4())
    local_output_path = os.path.abspath(os.path.join(output_dir, f"local_map_{job_id}.png"))

    try:
        exe_path = os.path.join(os.getcwd(), "run_task1.exe")

        cmd = [exe_path, local_rgb_path, local_nir_path, local_output_path]
        print(f"Running exe: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            error_message = f"Execution failed: {process.stderr or process.stdout}"
            print(f"ERROR: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)

        print("Local exe completed successfully.")
        return {
            "message": "Local image processing successful!",
            "output_saved_to": local_output_path
        }
    except Exception as e:
        print(f"Error during local processing: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 5. Main Supabase Endpoint ---
@app.post("/generate-stress-map/", dependencies=[Security(get_api_key)])
async def generate_map(request: ImageProcessRequest):
    BUCKET_NAME = "vegetation-maps"
    job_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), f"temp_{job_id}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        local_rgb_path = os.path.join(temp_dir, "input_rgb.png")
        local_nir_path = os.path.join(temp_dir, "input_nir.png")

        with open(local_rgb_path, 'wb+') as f:
            data = supabase.storage.from_(BUCKET_NAME).download(request.rgb_image_path)
            f.write(data)

        with open(local_nir_path, 'wb+') as f:
            data = supabase.storage.from_(BUCKET_NAME).download(request.nir_image_path)
            f.write(data)

        output_filename = f"stress_map_{job_id}.png"
        local_output_path = os.path.join(temp_dir, output_filename)

        exe_path = r"C:\Users\lundy\OneDrive\Documents\MATLAB\matlab_project\run_task1.exe"  # update path here too
        cmd = [exe_path, local_rgb_path, local_nir_path, local_output_path]
        print(f"Running exe: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            error_message = f"Execution failed: {process.stderr or process.stdout}"
            print(f"ERROR: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)

        print("Exe completed successfully.")

        output_storage_path = f"outputs/{output_filename}"
        with open(local_output_path, 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(file=f, path=output_storage_path, file_options={"content-type": "image/png"})

        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(output_storage_path)

        return {
            "message": "Stress map generated successfully!",
            "output_url": public_url
        }

    except Exception as e:
        print(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

# --- 6. Run FastAPI server ---
if __name__ == "__main__":
    uvicorn.run("RUNTASK:app", host="127.0.0.1", port=8000, reload=True)
