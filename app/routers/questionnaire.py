"""
Questionnaire Assessment Router
Handles psoriasis assessment form submissions and generates personalized recommendations using LLM.
"""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import uuid
import os
import json
import google.generativeai as genai

from app.schemas import QuestionnaireRequest, QuestionnaireResponse
from app.database import save_questionnaire_to_db

router = APIRouter()

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')


def generate_llm_assessment(data: QuestionnaireRequest) -> dict:
    """
    Use Gemini LLM to analyze psoriasis questionnaire and generate concise recommendations.
    
    Returns:
        dict: Contains severity, psa_risk, nextSteps (bullets), clinical_notes, urgency, followup_weeks
    """
    # Format questionnaire data for LLM
    patient_summary = f"""Patient Profile:
- Gender: {data.screen1.gender}, Age: {data.screen1.age}
- History: {data.screen1.psoriasisHistory}
- Affected areas: {', '.join(data.screen1.location)}
- Appearance: {', '.join(data.screen1.appearance)}
- Lesion size: {', '.join(data.screen1.size)}
- Nail changes: {', '.join(data.screen1.nails) if data.screen1.nails else 'None'}
- Scalp symptoms: {', '.join(data.screen1.scalp) if data.screen1.scalp else 'None'}

Symptom Details:
- Onset: {data.screen2.onsetDate}
- Pattern: {data.screen2.symptomPattern}
- Itching severity: {data.screen2.itching}/10
- Burning: {data.screen2.burning}/10
- Pain: {data.screen2.pain}/10
- Bleeding: {data.screen2.bleeding}/10
- Worsens at night: {data.screen2.worsenAtNight}
- Worsens with stress: {data.screen2.worsenWithStress}
- Triggers: {', '.join(data.screen2.triggers)}
- Medical triggers: {', '.join(data.screen2.medTriggers) if data.screen2.medTriggers else 'None'}
- Sunlight effect: {data.screen2.sunlightEffect}

Clinical Assessment:
- Daily impact: {data.screen3.dailyImpact}
- Emotional impact: {data.screen3.emotionalImpact}
- Joint pain: {data.screen3.jointPain}
- Joints affected: {', '.join(data.screen3.jointsAffected) if data.screen3.jointsAffected else 'None'}
- Nail changes with joint pain: {data.screen3.nailWithJoint}
- Past treatments: {data.screen3.pastTreatments}
- Current treatment: {data.screen3.currentTreatment}
- Treatment effectiveness: {data.screen3.reliefSideEffects}
- Family history: {', '.join(data.screen3.familyHistory)}
- Other conditions: {', '.join(data.screen3.otherConditions) if data.screen3.otherConditions else 'None'}
- Tried systemic therapy: {data.screen3.triedSystemic}
- Fever/infection: {data.screen3.feverInfection}
- Weight loss/fatigue: {data.screen3.weightLossFatigue}
"""

    prompt = f"""You are a dermatology AI assistant. Analyze this psoriasis questionnaire data.

{patient_summary}

Write a short, actionable plan in 3 sections. Use bullet points. Total length approx 150 words.

1. **Immediate Care**: Suggest specific care based on the high visual symptoms (e.g., if scaling is high, suggest keratolytics/moisturizers; if itch is high, suggest cooling/anti-itch).
2. **Trigger Management**: Give 1 specific tip for their listed triggers.
3. **Medical Strategy**: Analyze their current treatment. If their severity is high but they are only using mild treatments (or if they report it's not working), suggest the next class of treatment they should discuss with a doctor (e.g., Biologics, Phototherapy, or Systemics). *Do NOT mention specific dosage.*

*Important: End with a standard medical disclaimer that this is AI analysis, not a prescription.*

Also provide severity assessment, PSA risk, urgency, and follow-up timeline.

Respond ONLY with valid JSON in this exact format:
{{
  "severity": "mild|moderate|severe",
  "psa_risk": "low|medium|high",
  "urgency": "low|medium|high",
  "followup_weeks": 1-8,
  "nextSteps": ["bullet 1", "bullet 2", ...],
  "clinical_notes": "brief summary with the 3-section plan and medical disclaimer"
}}"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )
        
        # Parse Gemini response - extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        result = json.loads(response_text.strip())
        
        return {
            "severity": result.get("severity", "moderate"),
            "psa_risk": result.get("psa_risk", "medium"),
            "urgency": result.get("urgency", "medium"),
            "followup_weeks": result.get("followup_weeks", 4),
            "nextSteps": result.get("nextSteps", []),
            "clinical_notes": result.get("clinical_notes", "Assessment completed.")
        }
        
    except Exception as e:
        print(f"⚠️ LLM assessment failed, using fallback: {e}")
        # Fallback to simple rule-based assessment
        severity = "moderate"
        if len(data.screen1.location) >= 5:
            severity = "severe"
        elif len(data.screen1.location) <= 2 and data.screen2.itching < 5:
            severity = "mild"
        
        psa_risk = "medium" if data.screen3.jointPain == "yes" else "low"
        if data.screen3.jointPain == "yes" and data.screen3.nailWithJoint == "yes":
            psa_risk = "high"
        
        return {
            "severity": severity,
            "psa_risk": psa_risk,
            "urgency": "high" if severity == "severe" or psa_risk == "high" else "medium",
            "followup_weeks": 2 if severity == "severe" else 4,
            "nextSteps": [
                "Schedule dermatology consultation",
                "Continue current treatment regimen",
                "Monitor symptoms and document changes",
                "Maintain moisturizing routine",
                "Identify and avoid triggers"
            ],
            "clinical_notes": f"Assessment completed. {severity.capitalize()} severity psoriasis with {psa_risk} PSA risk. Follow-up recommended."
        }


@router.post("/submit", response_model=QuestionnaireResponse)
async def submit_questionnaire(
    request: QuestionnaireRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Submit psoriasis assessment questionnaire and receive personalized recommendations.
    
    This endpoint:
    1. Uses Gemini AI to assess psoriasis severity (mild/moderate/severe)
    2. Evaluates psoriatic arthritis risk (low/medium/high)
    3. Generates concise, personalized treatment recommendations
    4. Saves assessment to database with user tracking
    5. Returns actionable next steps and follow-up timeline
    
    Args:
        request: QuestionnaireRequest containing 3 screens of assessment data
        x_user_id: Optional user identifier from X-User-ID header
    
    Returns:
        QuestionnaireResponse with severity, risk assessment, and recommendations
    """
    try:
        # Generate unique assessment ID
        assessment_id = str(uuid.uuid4())
        
        # Use Gemini LLM to generate assessment and recommendations
        ai_recommendations = generate_llm_assessment(request)
        
        severity = ai_recommendations["severity"]
        psa_risk = ai_recommendations["psa_risk"]
        
        # Log assessment for monitoring
        print(f"\n{'='*60}")
        print(f"📋 New Questionnaire Assessment: {assessment_id}")
        print(f"👤 Patient: {request.screen1.gender}, Age {request.screen1.age}")
        print(f"📊 Severity: {severity.upper()}, PSA Risk: {psa_risk.upper()}")
        print(f"⏰ Urgency: {ai_recommendations['urgency'].upper()}")
        print(f"📅 Follow-up: {ai_recommendations['followup_weeks']} weeks")
        print(f"{'='*60}\n")
        
        # Prepare assessment data for database
        assessment_data = {
            "questionnaire_data": request.dict(),
            "severity": severity,
            "psa_risk": psa_risk,
            "recommendations": ai_recommendations["nextSteps"],
            "clinical_notes": ai_recommendations["clinical_notes"],
            "urgency": ai_recommendations["urgency"],
            "followup_weeks": ai_recommendations["followup_weeks"]
        }
        
        # Save to database
        db_metadata = save_questionnaire_to_db(
            assessment_data=assessment_data,
            assessment_id=assessment_id,
            user_id=x_user_id,
            timestamp=request.timestamp
        )
        
        # Return response
        return QuestionnaireResponse(
            assessment_id=assessment_id,
            timestamp=request.timestamp,
            severity_assessment=severity,
            psoriatic_arthritis_risk=psa_risk,
            nextSteps=ai_recommendations["nextSteps"],
            additionalNotes=ai_recommendations["clinical_notes"],
            treatment_urgency=ai_recommendations["urgency"],
            recommended_followup_weeks=ai_recommendations["followup_weeks"],
            db_metadata=db_metadata
        )
        
    except Exception as e:
        print(f"❌ Questionnaire processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Assessment processing failed: {str(e)}"
        )


@router.get("/history/{user_id}")
async def get_questionnaire_history(user_id: str, limit: int = 20):
    """
    Retrieve user's questionnaire assessment history.
    
    Args:
        user_id: User identifier
        limit: Maximum number of assessments to return (default: 20)
    
    Returns:
        List of past assessments with basic info
    """
    from app.database import get_user_assessments
    
    try:
        assessments = get_user_assessments(user_id, limit)
        return {
            "user_id": user_id,
            "count": len(assessments) if assessments else 0,
            "assessments": assessments or []
        }
    except Exception as e:
        print(f"❌ Failed to retrieve assessment history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{user_id}/{timestamp}")
async def get_questionnaire_result(user_id: str, timestamp: str):
    """
    Retrieve a specific questionnaire assessment by user_id and timestamp.
    
    Args:
        user_id: User identifier
        timestamp: ISO timestamp of the assessment
    
    Returns:
        Full assessment details
    """
    from app.database import get_assessment_by_timestamp
    
    try:
        result = get_assessment_by_timestamp(user_id, timestamp)
        if not result:
            raise HTTPException(status_code=404, detail="Assessment not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to retrieve assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
