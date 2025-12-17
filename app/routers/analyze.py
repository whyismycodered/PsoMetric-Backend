from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from app.services import ai_engine
from app.database import save_assessment, get_user_history, get_assessment
from app.llm_service import generate_recommendations
from PIL import Image
import io
import json
from typing import Optional

router = APIRouter()


@router.post("/")
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
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG, PNG, or WebP.")

    try:
        # Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. Parse questionnaire data
        questionnaire = {}
        if questionnaire_data:
            try:
                questionnaire = json.loads(questionnaire_data)
            except json.JSONDecodeError:
                print("⚠️ Failed to parse questionnaire_data JSON")
        
        # 2. Run ML Analysis
        ml_result = ai_engine.analyze_image(image)
        
        # Save to database
        db_result = save_analysis_to_db(
            analysis_result=result,
            user_id=user_id,
            image_filename=file.filename
        )
        
        # 5. Build unified response
        # Extract metrics from details if available
        erythema = 0
        induration = 0
        scaling = 0
        
        if ml_result.get("details"):
            # Average across all lesions
            details = ml_result["details"]
            if details:
                erythema = round(sum(d.get("erythema", 0) for d in details) / len(details), 1)
                induration = round(sum(d.get("induration", 0) for d in details) / len(details), 1)
                scaling = round(sum(d.get("desquamation", 0) for d in details) / len(details), 1)
        
        response = {
            # Assessment identifiers
            "assessment_id": db_info["assessment_id"] if db_info else None,
            "created_at": db_info["created_at"] if db_info else None,
            
            # ML Analysis Results
            "global_score": ml_result.get("global_score", 0),
            "diagnosis": ml_result.get("diagnosis", "Unknown"),
            "erythema": erythema,
            "induration": induration,
            "scaling": scaling,
            "lesions_found": ml_result.get("lesions_found", 0),
            "annotated_image_base64": ml_result.get("annotated_image_base64", ""),
            
            # LLM-Generated Recommendations
            "next_steps": llm_result.get("next_steps", []),
            "additional_notes": llm_result.get("additional_notes", ""),
            
            # Optional: detailed lesion data
            "details": ml_result.get("details", [])
        }
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@router.get("/history/{user_id}")
async def get_analysis_history(user_id: str, limit: int = 50):
    """
    Get all assessments for a user.
    
    Returns complete assessment records including:
    - ML results (scores, diagnosis)
    - LLM recommendations (next_steps, additional_notes)
    - Questionnaire data (for context)
    """
    try:
        assessments = get_user_history(user_id, limit)
        return {
            "user_id": user_id,
            "count": len(assessments),
            "assessments": assessments
        }
    except Exception as e:
        print(f"❌ History Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@router.get("/result/{user_id}/{created_at:path}")
async def get_assessment_result(user_id: str, created_at: str):
    """Get specific assessment by composite key."""
    try:
        result = get_assessment(user_id, created_at)
        if not result:
            raise HTTPException(status_code=404, detail="Assessment not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve assessment")