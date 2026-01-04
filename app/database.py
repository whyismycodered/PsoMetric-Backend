import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
from decimal import Decimal
import uuid

load_dotenv()

#comment#
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
            "annotated_image_url": ml_result.get("annotated_image_url", ""),
            
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

# ==================== QUESTIONNAIRE DATABASE FUNCTIONS ====================

def save_questionnaire_to_db(assessment_data: dict, assessment_id: str, user_id: str = None, timestamp: str = None):
    """
    Saves questionnaire assessment results to DynamoDB.
    
    Args:
        assessment_data: The assessment data including questionnaire, severity, recommendations
        assessment_id: Unique identifier for this assessment
        user_id: User identifier (required for DynamoDB key)
        timestamp: ISO timestamp from the request
    
    Returns:
        dict: Database metadata with assessment_id, timestamp, saved status or None on failure
    """
    table = get_dynamodb_table()
    if not table:
        print("⚠️ Database not available, skipping questionnaire save")
        return None
    
    # Default values
    if not user_id:
        user_id = 'anonymous'
    
    if not timestamp:
        timestamp = datetime.utcnow().isoformat()
    
    try:
        # Prepare record for DynamoDB - using flat questionnaire structure
        questionnaire_data = assessment_data.get('questionnaire_data', {})
        
        record = {
            'user_id': user_id,  # Partition key
            'created_at': timestamp,  # Sort key
            'assessment_id': assessment_id,
            'assessment_type': 'questionnaire',  # Distinguish from image analysis
            
            # Patient demographics
            'gender': questionnaire_data.get('gender'),
            'age': questionnaire_data.get('age'),
            
            # Assessment results
            'severity': assessment_data['severity'],
            'psa_risk': assessment_data['psa_risk'],
            'urgency': assessment_data['urgency'],
            'followup_weeks': assessment_data['followup_weeks'],
            
            # Clinical data
            'recommendations': assessment_data['recommendations'],
            'clinical_notes': assessment_data['clinical_notes'],
            
            # Full questionnaire data (for detailed analysis)
            'questionnaire_data': convert_to_decimal(questionnaire_data),
        }
        
        # Save to DynamoDB
        table.put_item(Item=record)
        print(f"✅ Questionnaire assessment saved to DB: {assessment_id} for user: {user_id}")
        
        return {
            'analysis_id': assessment_id,
            'timestamp': timestamp,
            'saved': True
        }
        
    except Exception as e:
        print(f"❌ Failed to save questionnaire to database: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_user_assessments(user_id: str, limit: int = 50):
    """
    Retrieves all assessments (both image and questionnaire) for a specific user.
    
    Args:
        user_id: The user identifier
        limit: Maximum number of records to return
    
    Returns:
        list: List of assessment records ordered by most recent
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
        
        # Return simplified view for history
        items = response.get('Items', [])
        return [{
            'assessment_id': item.get('assessment_id'),
            'created_at': item.get('created_at'),
            'assessment_type': item.get('assessment_type', 'image_analysis'),
            'severity': item.get('severity') or item.get('diagnosis'),
            'urgency': item.get('urgency', 'medium'),
        } for item in items]
        
    except Exception as e:
        print(f"❌ Failed to query user assessments: {e}")
        return []


def get_assessment_by_timestamp(user_id: str, timestamp: str):
    """
    Retrieves a specific assessment by user_id and timestamp (composite key).
    
    Args:
        user_id: The user identifier (partition key)
        timestamp: The creation timestamp (sort key)
    
    Returns:
        dict: Assessment record or None if not found
    """
    table = get_dynamodb_table()
    if not table:
        return None
    
    try:
        response = table.get_item(Key={
            'user_id': user_id,
            'created_at': timestamp
        })
        return response.get('Item')
    except Exception as e:
        print(f"❌ Failed to retrieve assessment: {e}")
        return None
