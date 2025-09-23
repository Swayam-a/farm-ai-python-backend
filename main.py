# ==============================================================================
# === Agro-Vision Companion: MATLAB Processing Backend =========================
# ==============================================================================
# This script creates a FastAPI server with two endpoints:
# 1. /process-local-images/: For easy testing with local files.
# 2. /generate-stress-map/: The main endpoint that works with Supabase.
# ==============================================================================

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

# --- 1. Configuration and Initialization ---
# Load all the secret keys and URLs from the .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SERVER_API_KEY = os.getenv("SERVER_API_KEY")

# Immediately check if all required environment variables are set
if not all([SUPABASE_URL, SUPABASE_KEY, SERVER_API_KEY]):
    raise ValueError("FATAL ERROR: Missing required environment variables. Please check your .env file.")

# Initialize the Supabase client and the FastAPI application
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI(title="Vegetation Health Mapping API")

# Define the name of the API key header for security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


# --- 2. API Key Authentication ---
async def get_api_key(api_key: str = Security(api_key_header)):
    """This function validates the API key sent in the request's header."""
    if api_key == SERVER_API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate API credentials.")


# --- 3. Request Data Model ---
class ImageProcessRequest(BaseModel):
    """Defines the structure of the JSON body for the Supabase endpoint."""
    rgb_image_path: str
    nir_image_path: str


# --- 4. Local Testing Endpoint ---
@app.post("/process-local-images/")
def process_local_images():
    """
    Processes images from a local 'local_test_data' folder.
    This is perfect for debugging your MATLAB script without needing Supabase.
    """
    print("--- Starting Local Processing Job ---")
    project_path = os.getcwd()
    local_data_dir = os.path.join(project_path, "local_test_data")
    output_dir = os.path.join(project_path, "local_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # IMPORTANT: We are now looking for .jpg files to match your test data
    local_rgb_path = os.path.abspath(os.path.join(local_data_dir, "test_rgb.jpg"))
    local_nir_path = os.path.abspath(os.path.join(local_data_dir, "test_nir.jpg"))
    
    # Check if the required test files exist
    if not os.path.exists(local_rgb_path) or not os.path.exists(local_nir_path):
        raise HTTPException(
            status_code=404, 
            detail="Test files not found. Make sure 'test_rgb.jpg' and 'test_nir.jpg' are in the 'local_test_data' folder."
        )

    job_id = str(uuid.uuid4())
    local_output_path = os.path.abspath(os.path.join(output_dir, f"local_map_{job_id}.png"))

    try:
        # Construct and execute the MATLAB command
        matlab_command = (
            f"matlab -batch \""
            f"cd '{project_path}'; "
            f"runner('{local_rgb_path}', '{local_nir_path}', '{local_output_path}')\""
        )
        print(f"Executing MATLAB command: {matlab_command}")
        process = subprocess.run(matlab_command, shell=True, capture_output=True, text=True)

        # Check if MATLAB ran successfully
        if process.returncode != 0:
            error_message = f"MATLAB execution failed: {process.stderr or process.stdout}"
            print(f"ERROR: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)

        print("MATLAB script completed successfully.")
        return {
            "message": "Local processing successful!",
            "output_saved_to": local_output_path
        }
    except Exception as e:
        # Catch any other unexpected errors and print a detailed traceback
        print(f"!!! AN UNEXPECTED ERROR OCCURRED: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 5. Main Supabase Endpoint ---
@app.post("/generate-stress-map/", dependencies=[Security(get_api_key)])
async def generate_map(request: ImageProcessRequest):
    """
    The main production endpoint. Downloads images from Supabase, processes
    them with MATLAB, and uploads the resulting stress map.
    """
    BUCKET_NAME = "vegetation-maps"  # Make sure this matches your Supabase bucket name
    job_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.getcwd(), f"temp_{job_id}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"--- Starting Supabase Processing Job {job_id} ---")

    try:
        # Step A: Download input images from Supabase
        print(f"Downloading images from bucket '{BUCKET_NAME}'...")
        local_rgb_path = os.path.join(temp_dir, "input_rgb.png")
        local_nir_path = os.path.join(temp_dir, "input_nir.png")

        with open(local_rgb_path, 'wb+') as f:
            res = supabase.storage.from_(BUCKET_NAME).download(request.rgb_image_path)
            f.write(res)
        with open(local_nir_path, 'wb+') as f:
            res = supabase.storage.from_(BUCKET_NAME).download(request.nir_image_path)
            f.write(res)
        
        # Step B: Prepare and execute MATLAB command
        output_filename = f"stress_map_{job_id}.png"
        local_output_path = os.path.join(temp_dir, output_filename)
        project_path = os.getcwd()
        matlab_command = (
            f"matlab -batch \""
            f"cd '{project_path}'; "
            f"runner('{os.path.abspath(local_rgb_path)}', '{os.path.abspath(local_nir_path)}', '{os.path.abspath(local_output_path)}')\""
        )
        print("Executing MATLAB script...")
        process = subprocess.run(matlab_command, shell=True, capture_output=True, text=True)

        if process.returncode != 0:
            error_message = f"MATLAB execution failed: {process.stderr or process.stdout}"
            print(f"ERROR: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
        print("MATLAB script completed successfully.")
        
        # Step C: Upload the resulting map back to Supabase
        output_storage_path = f"outputs/{output_filename}"
        print(f"Uploading result to: {output_storage_path}")
        with open(local_output_path, 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                file=f, path=output_storage_path, file_options={"content-type": "image/png"}
            )
        
        # Step D: Get the public URL and return it to the client
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(output_storage_path)
        return { "message": "Stress map generated successfully!", "output_url": public_url }
        
    except Exception as e:
        # This will print a very detailed error to your terminal for debugging
        print(f"!!! AN UNEXPECTED ERROR OCCURRED: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Step E: Always clean up the temporary files and folder
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")