"""
Test script for database integration
Run this to verify database connection and functionality
"""

from app.database import get_dynamodb_table, save_analysis_to_db, get_analysis_by_id, get_user_analyses

def test_connection():
    """Test DynamoDB connection"""
    print("🧪 Testing DynamoDB connection...")
    table = get_dynamodb_table()
    if table:
        print("✅ Successfully connected to DynamoDB")
        print(f"   Table name: {table.table_name}")
        return True
    else:
        print("❌ Failed to connect to DynamoDB")
        return False

def test_save():
    """Test saving analysis to database"""
    print("\n🧪 Testing save operation...")
    
    test_analysis = {
        "diagnosis": "Moderate",
        "global_score": 5.67,
        "lesions_found": 2,
        "details": [
            {
                "id": 1,
                "diagnosis": "Moderate",
                "severity_score": 6.25,
                "area_pixels": 12500,
                "erythema": 3,
                "induration": 2,
                "desquamation": 2
            },
            {
                "id": 2,
                "diagnosis": "Mild",
                "severity_score": 3.5,
                "area_pixels": 8000,
                "erythema": 2,
                "induration": 1,
                "desquamation": 1
            }
        ]
    }
    
    result = save_analysis_to_db(
        analysis_result=test_analysis,
        user_id="test_user",
        image_filename="test_image.jpg"
    )
    
    if result:
        print("✅ Successfully saved analysis to database")
        print(f"   Analysis ID: {result['analysis_id']}")
        print(f"   Timestamp: {result['timestamp']}")
        return ('test_user', result['timestamp'])
    else:
        print("❌ Failed to save analysis")
        return None

def test_retrieve(key_tuple):
    """Test retrieving analysis from database"""
    user_id, created_at = key_tuple
    print(f"\n🧪 Testing retrieval for user: {user_id}, created_at: {created_at}...")
    
    result = get_analysis_by_id(user_id, created_at)
    if result:
        print("✅ Successfully retrieved analysis")
        print(f"   Diagnosis: {result.get('diagnosis')}")
        print(f"   Global Score: {result.get('global_score')}")
        print(f"   Lesions Found: {result.get('lesions_found')}")
        return True
    else:
        print("❌ Failed to retrieve analysis")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Database Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Connection
    if not test_connection():
        print("\n⚠️ Cannot proceed with tests - database connection failed")
        print("   Please check your .env file and AWS credentials")
        exit(1)
    
    # Test 2: Save
    key_tuple = test_save()
    if not key_tuple:
        print("\n⚠️ Save operation failed")
        exit(1)
    
    # Test 3: Retrieve
    test_retrieve(key_tuple)
    
    # Test 4: User history
    user_id, _ = key_tuple
    results = get_user_analyses(user_id, limit=10)
    if results:
        print(f"\n✅ User history retrieved: {len(results)} record(s)")
    else:
        print("\n⚠️ No user history found")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

