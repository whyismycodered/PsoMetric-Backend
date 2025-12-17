import requests
import json
import os
from PIL import Image
import io

# Create a simple test image
img = Image.new('RGB', (224, 224), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

# Test questionnaire data
questionnaire = {
    'gender': 'Male',
    'age': 35,
    'psoriasisHistory': 'Yes',
    'location': ['scalp', 'elbows'],
    'appearance': ['red', 'scaly'],
    'size': ['small'],
    'itching': 6,
    'pain': 4,
    'jointPain': 'No',
    'jointsAffected': [],
    'dailyImpact': 'Moderate',
    'currentTreatment': 'Topical creams'
}

# Send request
files = {'file': ('test.png', img_bytes, 'image/png')}
data = {'questionnaire_data': json.dumps(questionnaire)}
headers = {'X-User-ID': 'test-user-db-check'}

print('📤 Sending test request...')
response = requests.post('http://127.0.0.1:8000/analyze/', files=files, data=data, headers=headers)

print(f'\n✅ Status: {response.status_code}')
if response.status_code == 200:
    result = response.json()
    print(f'\n📋 Response Structure:')
    print(f'  - Assessment ID: {result.get("assessment_id")}')
    print(f'  - Created At: {result.get("created_at")}')
    print(f'  - Diagnosis: {result.get("diagnosis")}')
    print(f'  - Global Score: {result.get("global_score")}')
    print(f'  - Erythema: {result.get("erythema")}')
    print(f'  - Induration: {result.get("induration")}')
    print(f'  - Scaling: {result.get("scaling")}')
    print(f'  - Lesions Found: {result.get("lesions_found")}')
    print(f'  - Next Steps: {len(result.get("next_steps", []))} items')
    print(f'  - Additional Notes: {result.get("additional_notes", "")[:50]}...')
    
    print(f'\n🔍 Next Steps:')
    for i, step in enumerate(result.get("next_steps", []), 1):
        print(f'  {i}. {step}')
    
    print(f'\n📚 Checking database history...')
    history = requests.get('http://127.0.0.1:8000/analyze/history/test-user-db-check')
    
    if history.status_code == 200:
        hist_data = history.json()
        print(f'  - Total assessments: {hist_data.get("count")}')
        
        if hist_data.get("assessments"):
            latest = hist_data["assessments"][0]
            print(f'\n✅ Database Storage Verified:')
            print(f'  - Has ML results: {all(k in latest for k in ["global_score", "diagnosis", "erythema"])}')
            print(f'  - Has LLM results: {all(k in latest for k in ["next_steps", "additional_notes"])}')
            print(f'  - Has questionnaire: {"questionnaire" in latest}')
            
            if "questionnaire" in latest:
                q = latest["questionnaire"]
                print(f'\n📝 Stored Questionnaire Sample:')
                print(f'  - Gender: {q.get("gender")}')
                print(f'  - Age: {q.get("age")}')
                print(f'  - Psoriasis History: {q.get("psoriasisHistory")}')
                print(f'  - Itching Level: {q.get("itching")}')
else:
    print(f'❌ Error: {response.text}')
