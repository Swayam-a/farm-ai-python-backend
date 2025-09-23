import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import cv2 # OpenCV for image processing
from sklearn.cluster import KMeans

app = FastAPI(title="Vegetation Health Mapping API")

# --- Image Processing Core Logic (No changes needed here) ---
def calculate_ndvi(rgb_image, nir_image):
    nir = nir_image.astype(np.float32)
    red = rgb_image[:, :, 2].astype(np.float32)
    np.seterr(divide='ignore', invalid='ignore')
    denominator = (nir + red)
    ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)
    return ndvi

def create_health_map(ndvi_map):
    pixels = ndvi_map.reshape((-1, 1))
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()
    sorted_centers_indices = np.argsort(centers)
    HEALTHY_COLOR, STRESSED_COLOR, HIGHLY_STRESSED_COLOR = [0, 255, 0], [0, 255, 255], [0, 0, 255]
    color_map = np.array([HIGHLY_STRESSED_COLOR, STRESSED_COLOR, HEALTHY_COLOR], dtype=np.uint8)
    segmented_image = color_map[np.argsort(sorted_centers_indices)[labels]].reshape(ndvi_map.shape + (3,))
    return segmented_image

# --- Main API Endpoint ---
@app.get("/generate-health-map", response_class=FileResponse)
async def generate_map():
    try:
        # Define paths to test images included in your project
        project_path = os.getcwd()
        local_data_dir = os.path.join(project_path, "local_test_data")
        
        rgb_path = os.path.join(local_data_dir, "test_rgb.jpg")
        nir_path = os.path.join(local_data_dir, "test_nir.jpg")
        output_path = os.path.join(project_path, "health_map_output.png")

        if not os.path.exists(rgb_path) or not os.path.exists(nir_path):
            raise HTTPException(status_code=404, detail="Test images not found on the server.")

        # Read images using OpenCV
        rgb_array = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        nir_array = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

        if rgb_array is None or nir_array is None:
            raise HTTPException(status_code=400, detail="Could not read test images.")

        # --- Run the Python-based processing pipeline ---
        ndvi_result = calculate_ndvi(rgb_array, nir_array)
        health_map_image = create_health_map(ndvi_result)

        # Save the final image to a file
        cv2.imwrite(output_path, health_map_image)
        
        # Return the generated image file
        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        print(f"An unexpected error occurred: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")