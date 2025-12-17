from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Query, Form
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
    questionnaire_data: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    user_id_query: Optional[str] = Query(None, alias="user_id")
):
    """
    Complete psoriasis analysis endpoint.
    
    Receives:
    - file: Lesion image (JPEG/PNG/WebP)
    - questionnaire_data: JSON string with questionnaire answers
    - X-User-ID header or user_id query param for database tracking
    
    Flow:
    1. Parse questionnaire from JSON string
    2. Run ML analysis on image (Sniper + Judge)
    3. Generate LLM recommendations using Gemini
    4. Save ONE complete record to DynamoDB
    5. Return unified response
    """
    user_id = x_user_id or user_id_query
    
    # Validate image format
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
        
        if ml_result.get("error"):
            raise HTTPException(status_code=400, detail=ml_result["error"])
        
        # 3. Generate LLM Recommendations
        llm_result = generate_recommendations(ml_result, questionnaire)
        
        # 4. Save to database (if user_id provided)
        db_info = None
        if user_id:
            db_info = save_assessment(
                user_id=user_id,
                ml_result=ml_result,
                llm_result=llm_result,
                questionnaire=questionnaire
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