import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
from decimal import Decimal
import uuid

# 1. Load the secret keys from .env
load_dotenv()

def get_dynamodb_table():
    """
    Creates a connection to the specific DynamoDB table.
    """
    try:
        # 2. Connect to AWS
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        # 3. Select the Table
        table = dynamodb.Table('PsoMetricDB')
        return table
        
    except Exception as e:
        print(f"❌ AWS Connection Error: {e}")
        return None

def convert_to_decimal(obj):
    """
    Recursively converts float values to Decimal for DynamoDB compatibility.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_decimal(item) for item in obj]
    return obj

def save_analysis_to_db(analysis_result: dict, user_id: str = None, image_filename: str = None):
    """
    Saves AI analysis results to DynamoDB.
    
    Args:
        analysis_result: The analysis result dictionary from AI engine
        user_id: User identifier (required for DynamoDB key)
        image_filename: Original filename of analyzed image
    
    Returns:
        dict: Saved record with analysis_id and created_at or None on failure
    """
    table = get_dynamodb_table()
    if not table:
        print("⚠️ Database not available, skipping save")
        return None
    
    # Require user_id since it's the partition key
    if not user_id:
        user_id = 'anonymous'
    
    try:
        # Generate unique analysis ID and timestamp
        analysis_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        # Prepare record for DynamoDB
        record = {
            'user_id': user_id,  # Partition key
            'created_at': created_at,  # Sort key
            'analysis_id': analysis_id,
            'image_filename': image_filename or 'unknown',
            
            # Global results
            'diagnosis': analysis_result.get('diagnosis'),
            'global_score': convert_to_decimal(analysis_result.get('global_score')),
            'lesions_found': analysis_result.get('lesions_found'),
            
            # Lesion details (convert all floats to Decimal)
            'lesion_details': convert_to_decimal(analysis_result.get('details', [])),
            
            # Metadata
            'has_annotated_image': analysis_result.get('annotated_image_base64') is not None,
        }
        
        # Save to DynamoDB
        table.put_item(Item=record)
        print(f"✅ Analysis saved to DB: {analysis_id} for user: {user_id}")
        
        return {
            'analysis_id': analysis_id,
            'timestamp': created_at,
            'saved': True
        }
        
    except Exception as e:
        print(f"❌ Failed to save to database: {e}")
        return None

def get_analysis_by_id(user_id: str, created_at: str):
    """
    Retrieves a specific analysis from the database using composite key.
    
    Args:
        user_id: The user identifier (partition key)
        created_at: The creation timestamp (sort key)
    
    Returns:
        dict: Analysis record or None if not found
    """
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
        print(f"❌ Failed to retrieve analysis: {e}")
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