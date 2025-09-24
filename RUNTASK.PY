# Forcing a new deployment at 5:28 AM
import io
# ... the rest of your code
import traceback
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image

# --- 1. Configuration ---
app = FastAPI(title="Lightweight Vegetation Health API")
MAX_DIMENSION = 1024

# --- 2. Lazy-Loaded Processing Class ---

# This class will hold all our processing logic.
# It won't be loaded into memory until get_pipeline() is called.
class ProcessingPipeline:
    def __init__(self):
        print("Processing pipeline initialized for the first time.")
        # In the future, if you had a model, you would load it here.
        # self.model = load_model("my_model.pkl")

    def calculate_ndvi(self, rgb_image_arr, nir_image_arr):
        """Calculates NDVI from resized RGB and NIR numpy arrays."""
        if rgb_image_arr.shape[:2] != nir_image_arr.shape[:2]:
            raise ValueError("RGB and NIR images must have the same dimensions.")

        nir = nir_image_arr.astype(np.float32)
        red = rgb_image_arr[:, :, 0].astype(np.float32)

        np.seterr(divide='ignore', invalid='ignore')
        denominator = (nir + red)
        ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)
        return ndvi

    def create_health_map_with_thresholding(self, ndvi_map):
        """Creates a color-coded health map using simple, fast thresholds."""
        HEALTHY_THRESHOLD = 0.4
        STRESSED_THRESHOLD = 0.15
        
        HEALTHY_COLOR = (0, 255, 0)
        STRESSED_COLOR = (255, 255, 0)
        HIGHLY_STRESSED_COLOR = (255, 0, 0)
        
        segmented_image = np.zeros((*ndvi_map.shape, 3), dtype=np.uint8)
        
        segmented_image[ndvi_map >= HEALTHY_THRESHOLD] = HEALTHY_COLOR
        segmented_image[(ndvi_map >= STRESSED_THRESHOLD) & (ndvi_map < HEALTHY_THRESHOLD)] = STRESSED_COLOR
        segmented_image[ndvi_map < STRESSED_THRESHOLD] = HIGHLY_STRESSED_COLOR
        
        return segmented_image

# --- 3. Global Placeholder & Loading Function ---

# Start with an empty placeholder. This uses almost no memory.
PIPELINE = None

def get_pipeline():
    """Loads the processing pipeline on the first request only."""
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = ProcessingPipeline()
    return PIPELINE

# --- 4. Main API Endpoint ---

@app.post("/generate-health-map/")
async def generate_map(rgb_file: UploadFile = File(...), nir_file: UploadFile = File(...)):
    """
    Accepts images, resizes them to save memory, creates a health map,
    and returns the result.
    """
    try:
        # Get the processing pipeline. The heavy loading only happens on the first call.
        pipeline = get_pipeline()

        # Read and prepare images
        rgb_contents = await rgb_file.read()
        nir_contents = await nir_file.read()

        rgb_pil = Image.open(io.BytesIO(rgb_contents)).convert("RGB")
        nir_pil = Image.open(io.BytesIO(nir_contents)).convert("L")

        rgb_pil.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
        nir_pil.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

        rgb_array = np.array(rgb_pil)
        nir_array = np.array(nir_pil)

        # Run the processing using the pipeline object
        ndvi_result = pipeline.calculate_ndvi(rgb_array, nir_array)
        health_map_array = pipeline.create_health_map_with_thresholding(ndvi_result)

        # Encode and return the final image
        health_map_pil = Image.fromarray(health_map_array)
        img_buffer = io.BytesIO()
        health_map_pil.save(img_buffer, "PNG")
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/png")

    except Exception as e:
        print(f"An unexpected error occurred: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")