# Database Integration Guide

## Overview
The PsoMetric Backend now automatically saves all AI analysis results to AWS DynamoDB after processing.

## Features
✅ Automatic saving after AI analysis  
✅ User history tracking  
✅ Analysis retrieval by ID  
✅ Timestamp-based ordering  
✅ Graceful failure handling (analysis continues even if DB save fails)

---

## DynamoDB Table Setup

### Table Configuration
- **Table Name:** `PsoMetricDB`
- **Primary Key:** `analysis_id` (String)
- **Region:** ap-southeast-2 (or your preferred region)

### Recommended Global Secondary Index (GSI)
For efficient user history queries:
- **Index Name:** `user_id-timestamp-index`
- **Partition Key:** `user_id` (String)
- **Sort Key:** `timestamp` (String)
- **Projection Type:** ALL

---

## Data Structure

### Stored Record Format
```json
{
  "analysis_id": "uuid-v4-string",
  "timestamp": "2025-12-07T10:30:45.123Z",
  "user_id": "user123",
  "image_filename": "lesion_photo.jpg",
  
  "diagnosis": "Moderate",
  "global_score": 5.67,
  "lesions_found": 2,
  
  "lesion_details": [
    {
      "id": 1,
      "diagnosis": "Moderate",
      "severity_score": 6.25,
      "area_pixels": 12500,
      "erythema": 3,
      "induration": 2,
      "desquamation": 2
    }
  ],
  
  "has_annotated_image": true
}
```

---

## API Endpoints

### 1. Analyze Image (with Auto-Save)
**POST** `/analyze/`

**Headers:**
- `X-User-ID` (optional): User identifier for tracking

**Body:**
- `file`: Image file (multipart/form-data)

**Response:**
```json
{
  "diagnosis": "Moderate",
  "global_score": 5.67,
  "lesions_found": 2,
  "annotated_image_base64": "base64_string...",
  "details": [...],
  "db_metadata": {
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-12-07T10:30:45.123Z",
    "saved": true
  }
}
```

### 2. Get Analysis History
**GET** `/analyze/history/{user_id}?limit=50`

**Parameters:**
- `user_id`: User identifier (path parameter)
- `limit`: Maximum results (query parameter, default: 50)

**Response:**
```json
{
  "user_id": "user123",
  "count": 15,
  "analyses": [...]
}
```

### 3. Get Specific Analysis
**GET** `/analyze/result/{analysis_id}`

**Parameters:**
- `analysis_id`: Unique analysis identifier

**Response:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-12-07T10:30:45.123Z",
  "diagnosis": "Moderate",
  ...
}
```

---

## Environment Configuration

### Required Environment Variables
Create a `.env` file in the project root:

```env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=ap-southeast-2
```

⚠️ **Security Note:** Never commit `.env` to version control!

---

## AWS IAM Permissions

### Required DynamoDB Permissions
Your AWS IAM user/role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": [
        "arn:aws:dynamodb:ap-southeast-2:*:table/PsoMetricDB",
        "arn:aws:dynamodb:ap-southeast-2:*:table/PsoMetricDB/index/*"
      ]
    }
  ]
}
```

---

## Usage Examples

### Python Client Example
```python
import requests

# Analyze image with user tracking
with open('lesion.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze/',
        files={'file': f},
        headers={'X-User-ID': 'user123'}
    )

result = response.json()
print(f"Analysis ID: {result['db_metadata']['analysis_id']}")
print(f"Diagnosis: {result['diagnosis']}")
```

### Get User History
```python
response = requests.get('http://localhost:8000/analyze/history/user123?limit=10')
history = response.json()
print(f"Found {history['count']} analyses")
```

### Retrieve Specific Analysis
```python
analysis_id = "550e8400-e29b-41d4-a716-446655440000"
response = requests.get(f'http://localhost:8000/analyze/result/{analysis_id}')
analysis = response.json()
```

---

## Error Handling

### Graceful Failure
If database connection fails:
- ⚠️ Warning logged to console
- ✅ Analysis continues normally
- ❌ `db_metadata` field will be `null` in response
- 👤 User still receives analysis results

### Database Connection Issues
Check logs for:
```
❌ AWS Connection Error: [error details]
⚠️ Database not available, skipping save
❌ Failed to save to database: [error details]
```

---

## Testing

### Test Database Connection
```python
from app.database import get_dynamodb_table

table = get_dynamodb_table()
if table:
    print("✅ Connected to DynamoDB")
else:
    print("❌ Connection failed")
```

### Test Save Function
```python
from app.database import save_analysis_to_db

test_result = {
    "diagnosis": "Mild",
    "global_score": 2.5,
    "lesions_found": 1,
    "details": []
}

db_result = save_analysis_to_db(
    analysis_result=test_result,
    user_id="test_user",
    image_filename="test.jpg"
)

if db_result:
    print(f"✅ Saved with ID: {db_result['analysis_id']}")
```

---

## Data Types

### DynamoDB Decimal Conversion
Python floats are automatically converted to DynamoDB Decimal type:
- `global_score`: 5.67 → Decimal('5.67')
- `severity_score`: 6.25 → Decimal('6.25')

This ensures precision and DynamoDB compatibility.

---

## Monitoring

### CloudWatch Metrics (Optional)
Monitor your DynamoDB table:
- Read/Write capacity units
- Throttled requests
- Item count
- Table size

### Application Logs
Watch for:
- `✅ Analysis saved to DB: [analysis_id]`
- `❌ Failed to save to database: [error]`
- `⚠️ Database not available, skipping save`

---

## Backup & Recovery

### DynamoDB Point-in-Time Recovery
Enable PITR for the PsoMetricDB table:
```bash
aws dynamodb update-continuous-backups \
    --table-name PsoMetricDB \
    --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true
```

### Export Data
```python
from app.database import get_dynamodb_table
import json

table = get_dynamodb_table()
response = table.scan()
items = response['Items']

with open('backup.json', 'w') as f:
    json.dump(items, f, indent=2, default=str)
```

---

## Cost Optimization

### On-Demand vs Provisioned Capacity
- **On-Demand:** Pay per request (recommended for variable traffic)
- **Provisioned:** Fixed capacity (cheaper for consistent traffic)

### GSI Considerations
- GSI doubles write costs (writes to both table and index)
- Only create GSI if frequently querying by user_id

---

## Troubleshooting

### Common Issues

1. **"AWS Connection Error"**
   - Check `.env` file exists and has correct credentials
   - Verify AWS region is correct
   - Test IAM permissions

2. **"Table not found"**
   - Create `PsoMetricDB` table in DynamoDB
   - Check table name spelling

3. **"ValidationException"**
   - Verify primary key structure
   - Check data types match schema

4. **Query returns empty results**
   - Verify GSI exists if querying by user_id
   - Check user_id value matches stored data
   - Try Scan as fallback (slower but works without GSI)

---

## Future Enhancements

- [ ] Store annotated images in S3 (reference in DynamoDB)
- [ ] Add time-series analysis for tracking progression
- [ ] Implement data retention policies
- [ ] Add batch analysis support
- [ ] Create analytics dashboard queries
- [ ] Implement data encryption at rest
- [ ] Add audit logging for compliance

---

## Support

For issues or questions:
- GitHub: whyismycodered/PsoMetric-Backend
- Check application logs for detailed error messages
- Verify AWS credentials and permissions
- Test database connection independently

---

**Last Updated:** December 7, 2025
