from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import ai_engine
from app.schemas import PsoriasisAnalysisResponse
from PIL import Image
import io

router = APIRouter()

@router.post("/", response_model=PsoriasisAnalysisResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload image and get full medical analysis.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run Engine
        return ai_engine.analyze_image(image)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Analysis Failed")