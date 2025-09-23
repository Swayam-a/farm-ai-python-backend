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

# --- 2. Image Processing Core Logic ---

def calculate_ndvi(rgb_image, nir_image):
    """Calculates the NDVI map from RGB and NIR image arrays."""
    # Ensure images have the same dimensions
    if rgb_image.shape[:2] != nir_image.shape[:2]:
        raise ValueError("RGB and NIR images must have the same height and width.")

    nir = nir_image.astype(np.float32)
    red = rgb_image[:, :, 2].astype(np.float32) # OpenCV uses BGR order

    # Prevent division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    denominator = (nir + red)
    ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)
    return ndvi

def create_health_map(ndvi_map):
    """Applies K-Means clustering and creates a color-coded health map."""
    # Reshape the 2D NDVI map into a 1D array of pixels
    pixels = ndvi_map.reshape((-1, 1))

    # --- FIX: Subsample pixels for faster clustering ---
    # Use a smaller sample for K-Means to avoid timeouts/memory errors.
    # 100,000 pixels is often enough for a good result and is very fast.
    max_pixels_for_clustering = 100000
    if pixels.shape[0] > max_pixels_for_clustering:
        # Create a random permutation of indices and take the first N
        indices = np.random.permutation(pixels.shape[0])[:max_pixels_for_clustering]
        sample_pixels = pixels[indices]
    else:
        sample_pixels = pixels

    # K-Means Clustering on the sample to find the 3 main NDVI value centers
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(sample_pixels)
    
    # Use the trained model to predict the cluster for ALL pixels
    labels = kmeans.predict(pixels)
    
    # Find which cluster corresponds to Healthy (highest NDVI), Stressed, etc.
    centers = kmeans.cluster_centers_.flatten()
    sorted_centers_indices = np.argsort(centers)
    
    # Define colors for each zone (in BGR format for OpenCV)
    HEALTHY_COLOR = [0, 255, 0]       # Green
    STRESSED_COLOR = [0, 255, 255]    # Yellow
    HIGHLY_STRESSED_COLOR = [0, 0, 255] # Red

    # Create a mapping from cluster index to color based on NDVI value (low to high)
    # The cluster with the lowest center is "Highly Stressed", highest is "Healthy"
    color_map_ordered = np.array([HIGHLY_STRESSED_COLOR, STRESSED_COLOR, HEALTHY_COLOR], dtype=np.uint8)
    
    # Create the final color-coded image
    # We need a map from the original kmeans label to the color
    label_to_color = np.empty_like(color_map_ordered)
    for i, idx in enumerate(sorted_centers_indices):
        label_to_color[idx] = color_map_ordered[i]
        
    segmented_image = label_to_color[labels].reshape(ndvi_map.shape + (3,))
    
    return segmented_image

# --- 3. Main API Endpoint ---

@app.post("/generate-health-map/")
async def generate_map(rgb_file: UploadFile = File(...), nir_file: UploadFile = File(...)):
    """
    Accepts RGB and NIR image uploads, processes them to create a health map,
    and returns the resulting image.
    """
    try:
        rgb_contents = await rgb_file.read()
        nir_contents = await nir_file.read()

        rgb_array = cv2.imdecode(np.frombuffer(rgb_contents, np.uint8), cv2.IMREAD_COLOR)
        nir_array = cv2.imdecode(np.frombuffer(nir_contents, np.uint8), cv2.IMREAD_GRAYSCALE)

        if rgb_array is None or nir_array is None:
            raise HTTPException(status_code=400, detail="Could not decode one or both images. Ensure they are valid image files.")

        # --- Run the processing pipeline ---
        ndvi_result = calculate_ndvi(rgb_array, nir_array)
        health_map_image = create_health_map(ndvi_result)

        # Encode the final image to PNG in memory
        is_success, img_encoded = cv2.imencode(".png", health_map_image)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode the resulting image.")
            
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

    except ValueError as ve: # Catch dimension mismatch error specifically
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during processing: {str(e)}")

# --- 4. Run FastAPI server for local testing ---
if __name__ == "__main__":
    # Ensure your filename is RUNTASK.py or change "RUNTASK:app" to "your_filename:app"
    uvicorn.run("RUNTASK:app", host="127.0.0.1", port=8000, reload=True)