import io
import traceback
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image

# --- 1. Configuration ---
app = FastAPI(title="Lightweight Vegetation Health API")

# Define a maximum size for images to prevent memory overload.
# Images will be resized to fit within a 1024x1024 box.
MAX_DIMENSION = 1024

# --- 2. Optimized Image Processing Logic ---

def calculate_ndvi(rgb_image_arr, nir_image_arr):
    """Calculates NDVI from resized RGB and NIR numpy arrays."""
    # Ensure images have the same dimensions (should be guaranteed by resizing)
    if rgb_image_arr.shape[:2] != nir_image_arr.shape[:2]:
        raise ValueError("RGB and NIR images must have the same dimensions.")

    # Convert to float for calculation
    nir = nir_image_arr.astype(np.float32)
    red = rgb_image_arr[:, :, 0].astype(np.float32) # In Pillow, Red is the first channel

    # Prevent division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    denominator = (nir + red)
    ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)
    return ndvi

def create_health_map_with_thresholding(ndvi_map):
    """Creates a color-coded health map using simple, fast thresholds."""
    # Define NDVI thresholds for different health levels
    # You can adjust these values based on your specific crops/environment
    HEALTHY_THRESHOLD = 0.4
    STRESSED_THRESHOLD = 0.15

    # Define colors (Pillow uses RGB format)
    HEALTHY_COLOR = (0, 255, 0)       # Green
    STRESSED_COLOR = (255, 255, 0)    # Yellow
    HIGHLY_STRESSED_COLOR = (255, 0, 0) # Red
    
    # Create an empty 3-channel image to store the colored result
    segmented_image = np.zeros((*ndvi_map.shape, 3), dtype=np.uint8)
    
    # Apply colors based on thresholds using efficient boolean masking
    segmented_image[ndvi_map >= HEALTHY_THRESHOLD] = HEALTHY_COLOR
    segmented_image[(ndvi_map >= STRESSED_THRESHOLD) & (ndvi_map < HEALTHY_THRESHOLD)] = STRESSED_COLOR
    segmented_image[ndvi_map < STRESSED_THRESHOLD] = HIGHLY_STRESSED_COLOR
    
    return segmented_image

# --- 3. Main API Endpoint ---

@app.post("/generate-health-map/")
async def generate_map(rgb_file: UploadFile = File(...), nir_file: UploadFile = File(...)):
    """
    Accepts images, resizes them to save memory, creates a health map
    using lightweight thresholding, and returns the result.
    """
    try:
        # Read image bytes
        rgb_contents = await rgb_file.read()
        nir_contents = await nir_file.read()

        # Open images with Pillow
        rgb_pil = Image.open(io.BytesIO(rgb_contents)).convert("RGB")
        nir_pil = Image.open(io.BytesIO(nir_contents)).convert("L") # L is for grayscale

        # --- MEMORY OPTIMIZATION: Resize images ---
        rgb_pil.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
        nir_pil.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

        # Convert resized images to numpy arrays
        rgb_array = np.array(rgb_pil)
        nir_array = np.array(nir_pil)

        # --- Run the lightweight processing pipeline ---
        ndvi_result = calculate_ndvi(rgb_array, nir_array)
        health_map_array = create_health_map_with_thresholding(ndvi_result)

        # Convert final result back to a Pillow image
        health_map_pil = Image.fromarray(health_map_array)
        
        # Save the image to an in-memory buffer
        img_buffer = io.BytesIO()
        health_map_pil.save(img_buffer, "PNG")
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/png")

    except Exception as e:
        print(f"An unexpected error occurred: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- 4. Local Server (Optional) ---
# This part is for local testing and does not run in production on Railway
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)