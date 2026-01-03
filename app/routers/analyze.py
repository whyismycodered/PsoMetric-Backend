from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services import ai_engine
from app.database import save_assessment, get_user_history, get_assessment
from app.llm_service import generate_recommendations
from PIL import Image
import io
import json

router = APIRouter()


@router.post("/")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    questionnaire_data: str = Form(...)
):
    """
    Complete psoriasis analysis endpoint.
    
    Receives:
    - file: Lesion image (JPEG/PNG/WebP)
    - questionnaire_data: JSON string with questionnaire answers (includes userId)
    
    Flow:
    1. Parse questionnaire from JSON string
    2. Run ML analysis on image (Sniper + Judge)
    3. Generate LLM recommendations using Gemini
    4. Save ONE complete record to DynamoDB
    5. Return unified response
    """
    # Validate image format
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG, PNG, or WebP.")

    try:
        # Step 1: Parse questionnaire data
        try:
            questionnaire = json.loads(questionnaire_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid questionnaire_data JSON")
        
        # Extract user_id from questionnaire
        user_id = questionnaire.get("userId")
        if not user_id:
            raise HTTPException(status_code=400, detail="userId is required in questionnaire_data")
        
        # Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Step 2: Run ML Analysis
        ml_result = ai_engine.analyze_image(image)
        
        if ml_result.get("error"):
            raise HTTPException(status_code=400, detail=ml_result["error"])
        
        # Extract individual scores from details (average across lesions)
        erythema = 0
        induration = 0
        scaling = 0
        
        if ml_result.get("details"):
            details = ml_result["details"]
            if details:
                erythema = round(sum(d.get("erythema", 0) for d in details) / len(details), 1)
                induration = round(sum(d.get("induration", 0) for d in details) / len(details), 1)
                scaling = round(sum(d.get("desquamation", 0) for d in details) / len(details), 1)
        
        # Add extracted scores to ml_result for database storage
        ml_result["erythema"] = erythema
        ml_result["induration"] = induration
        ml_result["scaling"] = scaling
        
        # Step 3: Generate LLM Recommendations
        llm_result = generate_recommendations(ml_result, questionnaire)
        
        # Step 4: Save to DynamoDB (single record with everything)
        db_result = save_assessment(
            user_id=user_id,
            ml_result=ml_result,
            llm_result=llm_result,
            questionnaire=questionnaire
        )
        
        if not db_result:
            print("⚠️ Failed to save to database, but continuing with response")
        
        # Step 5: Return complete response
        return {
            # Assessment identifiers
            "assessment_id": db_result["assessment_id"] if db_result else None,
            "created_at": db_result["created_at"] if db_result else None,
            
            # ML Analysis Results
            "global_score": ml_result.get("global_score", 0),
            "diagnosis": ml_result.get("diagnosis", "Unknown"),
            "erythema": erythema,
            "induration": induration,
            "scaling": scaling,
            "lesions_found": ml_result.get("lesions_found", 0),
            "annotated_image_url": ml_result.get("annotated_image_url", ""),
            
            # LLM-Generated Recommendations
            "next_steps": llm_result.get("next_steps", []),
            "additional_notes": llm_result.get("additional_notes", "")
        }

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