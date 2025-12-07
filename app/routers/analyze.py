from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from app.services import ai_engine
from app.schemas import PsoriasisAnalysisResponse, DatabaseMetadata
from app.database import save_analysis_to_db, get_analysis_by_id, get_user_analyses
from PIL import Image
import io
from typing import Optional

router = APIRouter()

@router.post("/", response_model=PsoriasisAnalysisResponse)
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Endpoint to upload image and get full medical analysis.
    Results are automatically saved to the database.
    
    Headers:
        X-User-ID (optional): User identifier for tracking analyses
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run AI Engine
        result = ai_engine.analyze_image(image)
        
        # Save to database
        db_result = save_analysis_to_db(
            analysis_result=result,
            user_id=user_id,
            image_filename=file.filename
        )
        
        # Add database metadata to response
        if db_result:
            result['db_metadata'] = DatabaseMetadata(
                analysis_id=db_result['analysis_id'],
                timestamp=db_result['timestamp'],
                saved=db_result['saved']
            )
        
        return result

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Analysis Failed")

@router.get("/history/{user_id}")
async def get_analysis_history(user_id: str, limit: int = 50):
    """
    Retrieve analysis history for a specific user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of records to return (default: 50)
    
    Returns:
        List of previous analyses ordered by created_at (most recent first)
    """
    try:
        analyses = get_user_analyses(user_id, limit)
        return {
            "user_id": user_id,
            "count": len(analyses),
            "analyses": analyses
        }
    except Exception as e:
        print(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

@router.get("/result/{user_id}/{created_at}")
async def get_analysis_result(user_id: str, created_at: str):
    """
    Retrieve a specific analysis result by composite key.
    
    Args:
        user_id: User identifier (partition key)
        created_at: Creation timestamp (sort key, ISO format)
    
    Returns:
        Analysis record
    """
    try:
        result = get_analysis_by_id(user_id, created_at)
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")