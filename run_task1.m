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

# Load environment variables
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

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == SERVER_API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate API credentials.")

class ImageProcessRequest(BaseModel):
    rgb_image_path: str
    nir_image_path: str

@app.post("/generate-stress-map/", dependencies=[Security(get_api_key)])
async def generate_map(request: ImageProcessRequest):
    BUCKET_NAME = "vegetation-maps"
    job_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), f"temp_{job_id}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Download images from Supabase storage
        local_rgb_path = os.path.join(temp_dir, "input_rgb.png")
        local_nir_path = os.path.join(temp_dir, "input_nir.png")

        with open(local_rgb_path, 'wb+') as f:
            data = supabase.storage.from_(BUCKET_NAME).download(request.rgb_image_path)
            f.write(data)

        with open(local_nir_path, 'wb+') as f:
            data = supabase.storage.from_(BUCKET_NAME).download(request.nir_image_path)
            f.write(data)

        # Prepare output path
        output_filename = f"stress_map_{job_id}.png"
        local_output_path = os.path.join(temp_dir, output_filename)

        # Path to your compiled executable
        exe_path = r"C:\Users\lundy\OneDrive\Documents\MATLAB\matlab_project\run_task1.exe"

        # Call the exe with input and output file paths
        cmd = [exe_path, local_rgb_path, local_nir_path, local_output_path]
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Execution failed: {process.stderr or process.stdout}")

        # Upload the result back to Supabase
        output_storage_path = f"outputs/{output_filename}"
        with open(local_output_path, 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(file=f, path=output_storage_path, file_options={"content-type": "image/png"})

        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(output_storage_path)

        return {
            "message": "Stress map generated successfully!",
            "output_url": public_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
