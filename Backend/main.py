from fastapi import FastAPI, UploadFile, File
import os
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "plant_model.h5")

print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully!")

# --- Class Names ---
class_names = [
    "Bell_Pepper_Bacterial_spot",
    "Bell_Pepper_healthy",
    "Strawberry_Leaf_Scorch",
    "Strawberry_healthy"
]

# --- Recommendations ---
recommendations = {
    "Bell_Pepper_Bacterial_spot": "Remove infected leaves and apply copper-based fungicide.",
    "Strawberry_Leaf_Scorch": "Improve air circulation and avoid overhead watering.",
    "Bell_Pepper_healthy": "Your plant looks healthy. Maintain proper watering and sunlight.",
    "Strawberry_healthy": "Your plant is healthy. Continue regular care."
}

# --- Test Route ---
@app.get("/")
def home():
    return {"message": "LexiGrow backend is running"}

# --- Prediction Route ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize image for model
        image = image.resize((224, 224))

        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        result = class_names[predicted_class]

        plant_name = result.rsplit("_", 1)[0]

        status = "Healthy" if "healthy" in result.lower() else "Diseased"

        return {
            "plant": plant_name,
            "disease": result,
            "confidence": round(confidence * 100, 2),
            "status": status,
            "recommendation": recommendations.get(result, "No recommendation available")
        }

    except Exception as e:
        return {"error": str(e)}