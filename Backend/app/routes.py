from fastapi import APIRouter, UploadFile, File
from app.preprocess import preprocess_image

router = APIRouter()

@router.get("/")
def home():
    return {"message": "LexiGrow backend alive"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed = preprocess_image(image_bytes)

    return {
        "message": "Image processed successfully",
        "shape": str(processed.shape)
    }