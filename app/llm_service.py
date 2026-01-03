"""
LLM Service for generating psoriasis recommendations using Google Gemini.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def generate_recommendations(ml_result: dict, questionnaire: dict) -> dict:
    """
    Generate personalized next steps and additional notes using Gemini.
    
    Args:
        ml_result: ML analysis output (global_score, diagnosis, erythema, etc.)
        questionnaire: User's questionnaire answers
    
    Returns:
        dict with 'next_steps' (list of strings) and 'additional_notes' (string)
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Build the prompt with context
        prompt = _build_prompt(ml_result, questionnaire)
        
        response = model.generate_content(prompt)
        
        # Parse the response
        return _parse_response(response.text)
        
    except Exception as e:
        print(f"⚠️ LLM generation failed: {e}")
        # Return fallback recommendations
        return _get_fallback_recommendations(ml_result)


def _build_prompt(ml_result: dict, questionnaire: dict) -> str:
    """Build the prompt for Gemini with all context."""
    
    diagnosis = ml_result.get("diagnosis", "Unknown")
    global_score = ml_result.get("global_score", 0)
    erythema = ml_result.get("erythema", 0)
    induration = ml_result.get("induration", 0)
    scaling = ml_result.get("desquamation", ml_result.get("scaling", 0))  # Handle both names
    lesions_found = ml_result.get("lesions_found", 0)
    
    # Questionnaire data
    age = questionnaire.get("age", "Unknown")
    gender = questionnaire.get("gender", "Unknown")
    psoriasis_history = questionnaire.get("psoriasisHistory", "Unknown")
    locations = questionnaire.get("location", [])
    appearance = questionnaire.get("appearance", [])
    size = questionnaire.get("size", [])
    itching = questionnaire.get("itching", 0)
    pain = questionnaire.get("pain", 0)
    joint_pain = questionnaire.get("jointPain", "No")
    joints_affected = questionnaire.get("jointsAffected", [])
    daily_impact = questionnaire.get("dailyImpact", "Unknown")
    current_treatment = questionnaire.get("currentTreatment", "None")
    
    prompt = f"""You are a dermatology assistant AI. Based on the following psoriasis analysis results and patient information, provide personalized recommendations.

## ML Analysis Results:
- Diagnosis: {diagnosis}
- Global Severity Score: {global_score}/10
- Erythema (redness): {erythema}/4
- Induration (thickness): {induration}/4
- Scaling: {scaling}/4
- Lesions detected: {lesions_found}

## Patient Information:
- Age: {age}
- Gender: {gender}
- Psoriasis history: {psoriasis_history}
- Affected body locations: {', '.join(locations) if locations else 'Not specified'}
- Lesion appearance: {', '.join(appearance) if appearance else 'Not specified'}
- Lesion size: {', '.join(size) if size else 'Not specified'}
- Itching level: {itching}/10
- Pain level: {pain}/10
- Joint pain: {joint_pain}
- Joints affected: {', '.join(joints_affected) if joints_affected else 'None'}
- Daily life impact: {daily_impact}
- Current treatment: {current_treatment}

## Task:
Provide practical, actionable recommendations in the following JSON format:
{{
    "next_steps": [
        "Step 1...",
        "Step 2...",
        "Step 3...",
        "Step 4..."
    ],
    "additional_notes": "Brief summary note about the condition and general advice."
}}

Guidelines:
- Provide 3-5 specific, actionable next steps
- Consider the severity level when making recommendations
- If joint pain is present, mention consulting a rheumatologist
- If symptoms are severe, recommend seeing a dermatologist soon
- Include skincare and lifestyle tips appropriate to severity
- Keep recommendations practical and patient-friendly
- IMPORTANT: Never diagnose - always recommend consulting a healthcare professional for diagnosis

Return ONLY the JSON object, no other text."""

    return prompt


def _parse_response(response_text: str) -> dict:
    """Parse Gemini's response into structured data."""
    import json
    
    try:
        # Clean up the response - remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        result = json.loads(text)
        
        # Ensure we have the expected structure
        return {
            "next_steps": result.get("next_steps", []),
            "additional_notes": result.get("additional_notes", "")
        }
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse LLM response: {e}")
        print(f"Response was: {response_text[:500]}")
        return {
            "next_steps": ["Please consult with a dermatologist for personalized recommendations."],
            "additional_notes": "Analysis complete. Professional consultation recommended."
        }


def _get_fallback_recommendations(ml_result: dict) -> dict:
    """Generate fallback recommendations when LLM is unavailable."""
    
    diagnosis = ml_result.get("diagnosis", "Unknown")
    global_score = ml_result.get("global_score", 0)
    
    if diagnosis == "Clear" or global_score < 1:
        return {
            "next_steps": [
                "Continue your current skincare routine",
                "Keep skin moisturized with fragrance-free products",
                "Monitor for any changes in skin condition",
                "Schedule routine check-ups with your dermatologist"
            ],
            "additional_notes": "No significant psoriasis activity detected. Continue preventive care."
        }
    elif diagnosis == "Mild" or global_score < 4:
        return {
            "next_steps": [
                "Apply prescribed topical treatments as directed",
                "Keep affected areas moisturized",
                "Avoid triggers like stress and skin injuries",
                "Consider scheduling a dermatology follow-up",
                "Track symptoms to identify patterns"
            ],
            "additional_notes": "Mild psoriasis detected. Topical treatments and lifestyle modifications may help manage symptoms."
        }
    elif diagnosis == "Moderate" or global_score < 7.5:
        return {
            "next_steps": [
                "Consult your dermatologist about current treatment effectiveness",
                "Discuss potential systemic treatment options",
                "Continue topical treatments as prescribed",
                "Consider phototherapy if recommended by your doctor",
                "Monitor for signs of psoriatic arthritis"
            ],
            "additional_notes": "Moderate psoriasis detected. Professional evaluation recommended to optimize treatment plan."
        }
    else:
        return {
            "next_steps": [
                "Schedule an urgent appointment with your dermatologist",
                "Discuss biologic or systemic treatment options",
                "Monitor for joint pain or swelling",
                "Consider consulting a rheumatologist if joint symptoms present",
                "Document symptoms and triggers for your healthcare team"
            ],
            "additional_notes": "Severe psoriasis detected. Prompt medical attention recommended. Advanced treatments may be beneficial."
        }
