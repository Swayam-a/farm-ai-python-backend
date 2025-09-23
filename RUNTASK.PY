import os
import io
import traceback
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import uvicorn
import numpy as np
import cv2 # OpenCV for image processing
from sklearn.cluster import KMeans

# --- 1. Configuration and Initialization ---
load_dotenv()
app = FastAPI(title="Vegetation Health Mapping API")

# --- 2. Image Processing Core Logic (Replaces MATLAB) ---

def calculate_ndvi(rgb_image, nir_image):
    """Calculates the NDVI map from RGB and NIR image arrays."""
    # The NIR band is a single channel (grayscale)
    nir = nir_image.astype(np.float32)
    # The Red band is the last channel in the RGB image when read by OpenCV (BGR order)
    red = rgb_image[:, :, 2].astype(np.float32)

    # Prevent division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    denominator = (nir + red)
    ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)
    return ndvi

def create_health_map(ndvi_map):
    """Applies K-Means clustering and creates a color-coded health map."""
    # Reshape the 2D NDVI map into a 1D array of pixels for clustering
    pixels = ndvi_map.reshape((-1, 1))

    # K-Means Clustering to identify 3 zones (Healthy, Stressed, Highly Stressed)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(pixels)
    labels = kmeans.labels_
    
    # Find which cluster corresponds to Healthy (highest NDVI), Stressed, etc.
    centers = kmeans.cluster_centers_.flatten()
    sorted_centers_indices = np.argsort(centers)
    
    # Define colors for each zone (in BGR format for OpenCV)
    # You can adjust these colors to your preference
    HEALTHY_COLOR = [0, 255, 0]       # Green
    STRESSED_COLOR = [0, 255, 255]    # Yellow
    HIGHLY_STRESSED_COLOR = [0, 0, 255] # Red

    # Create a mapping from sorted cluster index to color
    color_map = np.array([HIGHLY_STRESSED_COLOR, STRESSED_COLOR, HEALTHY_COLOR], dtype=np.uint8)
    
    # Create the final color-coded image
    segmented_image = color_map[np.argsort(sorted_centers_indices)[labels]].reshape(ndvi_map.shape + (3,))
    
    return segmented_image

# --- 3. Main API Endpoint ---

@app.post("/generate-health-map/")
async def generate_map(rgb_file: UploadFile = File(...), nir_file: UploadFile = File(...)):
    """
    Accepts RGB and NIR image uploads, processes them to create a health map,
    and returns the resulting image.
    """
    try:
        # Read the uploaded image files into memory
        rgb_contents = await rgb_file.read()
        nir_contents = await nir_file.read()

        # Convert the in-memory image data into NumPy arrays using OpenCV
        rgb_array = cv2.imdecode(np.frombuffer(rgb_contents, np.uint8), cv2.IMREAD_COLOR)
        nir_array = cv2.imdecode(np.frombuffer(nir_contents, np.uint8), cv2.IMREAD_GRAYSCALE)

        if rgb_array is None or nir_array is None:
            raise HTTPException(status_code=400, detail="Could not decode one or both of the uploaded images.")

        # --- Run the Python-based processing pipeline ---
        # 1. Calculate the NDVI map
        ndvi_result = calculate_ndvi(rgb_array, nir_array)
        
        # 2. Create the final color-coded health map image
        health_map_image = create_health_map(ndvi_result)

        # Encode the final image from a NumPy array to a PNG format in memory
        _, img_encoded = cv2.imencode(".png", health_map_image)
        
        # Return the in-memory image data as a streaming response
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

    except Exception as e:
        print(f"An unexpected error occurred: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")

# --- 4. Run FastAPI server for local testing ---
if __name__ == "__main__":
    uvicorn.run("RUNTASK:app", host="127.0.0.1", port=8000, reload=True)