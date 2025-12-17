import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
from decimal import Decimal
import uuid

load_dotenv()


def get_dynamodb_table():
    """Get DynamoDB table connection."""
    try:
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        return dynamodb.Table('PsoMetricDB')
    except Exception as e:
        print(f"❌ AWS Connection Error: {e}")
        return None


def convert_to_decimal(value):
    """Convert floats to Decimal for DynamoDB."""
    if isinstance(value, float):
        return Decimal(str(value))
    elif isinstance(value, dict):
        return {k: convert_to_decimal(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_decimal(v) for v in value]
    return value


def save_assessment(
    user_id: str,
    ml_result: dict,
    llm_result: dict,
    questionnaire: dict
) -> dict:
    """
    Save ONE complete assessment record containing:
    - ML analysis results (scores, diagnosis, annotated image)
    - LLM-generated recommendations (next_steps, additional_notes)
    - Questionnaire data (for history recall)
    
    Args:
        user_id: User identifier (partition key)
        ml_result: ML model output (global_score, diagnosis, erythema, etc.)
        llm_result: LLM output (next_steps, additional_notes)
        questionnaire: Original questionnaire answers
    
    Returns:
        dict with assessment_id, created_at, saved status
    """
    table = get_dynamodb_table()
    if not table:
        print("⚠️ Database not available")
        return None
    
    try:
        assessment_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        # Build the complete record
        record = {
            # Primary keys
            "user_id": user_id,
            "created_at": created_at,
            "assessment_id": assessment_id,
            
            # ML Analysis Results
            "global_score": convert_to_decimal(ml_result.get("global_score", 0)),
            "diagnosis": ml_result.get("diagnosis", "Unknown"),
            "erythema": convert_to_decimal(ml_result.get("erythema", 0)),
            "induration": convert_to_decimal(ml_result.get("induration", 0)),
            "scaling": convert_to_decimal(ml_result.get("scaling", 0)),
            "lesions_found": ml_result.get("lesions_found", 0),
            "annotated_image_base64": ml_result.get("annotated_image_base64", ""),
            
            # LLM-Generated Recommendations
            "next_steps": llm_result.get("next_steps", []),
            "additional_notes": llm_result.get("additional_notes", ""),
            
            # Questionnaire Data (for history recall)
            "questionnaire": convert_to_decimal({
                "gender": questionnaire.get("gender"),
                "age": questionnaire.get("age"),
                "psoriasisHistory": questionnaire.get("psoriasisHistory"),
                "location": questionnaire.get("location", []),
                "appearance": questionnaire.get("appearance", []),
                "size": questionnaire.get("size", []),
                "itching": questionnaire.get("itching", 0),
                "pain": questionnaire.get("pain", 0),
                "jointPain": questionnaire.get("jointPain"),
                "jointsAffected": questionnaire.get("jointsAffected", []),
                "dailyImpact": questionnaire.get("dailyImpact"),
                "currentTreatment": questionnaire.get("currentTreatment"),
            })
        }
        
        table.put_item(Item=record)
        print(f"✅ Assessment saved: {assessment_id} for user: {user_id}")
        
        return {
            "assessment_id": assessment_id,
            "created_at": created_at,
            "saved": True
        }
        
    except Exception as e:
        print(f"❌ Failed to save assessment: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_user_history(user_id: str, limit: int = 50):
    """Get all assessments for user."""
    table = get_dynamodb_table()
    if not table:
        return []
    
    try:
        response = table.query(
            KeyConditionExpression='user_id = :uid',
            ExpressionAttributeValues={':uid': user_id},
            ScanIndexForward=False,
            Limit=limit
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return []


def get_assessment(user_id: str, created_at: str):
    """Get specific assessment by composite key."""
    table = get_dynamodb_table()
    if not table:
        return None
    
    try:
        response = table.get_item(Key={
            'user_id': user_id,
            'created_at': created_at
        })
        return response.get('Item')
    except Exception as e:
        print(f"❌ Failed to get assessment: {e}")
        return None

def get_user_analyses(user_id: str, limit: int = 50):
    """
    Retrieves all analyses for a specific user, ordered by created_at (most recent first).
    
    Args:
        user_id: The user identifier
        limit: Maximum number of records to return
    
    Returns:
        list: List of analysis records
    """
    table = get_dynamodb_table()
    if not table:
        return []
    
    try:
        response = table.query(
            KeyConditionExpression='user_id = :uid',
            ExpressionAttributeValues={':uid': user_id},
            ScanIndexForward=False,  # Most recent first
            Limit=limit
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"❌ Failed to query user analyses: {e}")
        return []