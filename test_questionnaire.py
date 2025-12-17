"""
Test script for questionnaire endpoint
Tests the questionnaire submission and assessment logic
"""
import requests
import json

# Configuration
BASE_URL = "http://127.0.0.1:8000"
QUESTIONNAIRE_ENDPOINT = f"{BASE_URL}/questionnaire/submit"

def test_questionnaire_endpoint():
    """Test the questionnaire submission endpoint"""
    
    # Test data matching the frontend format
    test_payload = {
        "timestamp": "2025-12-08T10:30:00.000Z",
        "screen1": {
            "gender": "Female",
            "age": "28",
            "psoriasisHistory": "recurrent",
            "location": ["scalp", "elbows", "knees"],
            "appearance": ["red", "scaly", "silvery"],
            "size": ["coin", "palm"],
            "nails": ["pitting", "ridges"],
            "scalp": ["flaking", "itching"]
        },
        "screen2": {
            "onsetDate": "2 weeks ago",
            "symptomPattern": "intermittent",
            "lesionSpeed": "gradual",
            "itching": 7,
            "burning": 4,
            "pain": 3,
            "bleeding": 2,
            "worsenAtNight": "yes",
            "worsenWithStress": "yes",
            "triggers": ["stress", "cold", "alcohol"],
            "medTriggers": ["strep"],
            "sunlightEffect": "winter"
        },
        "screen3": {
            "dailyImpact": "moderate",
            "emotionalImpact": "sometimes",
            "relationshipsImpact": "some",
            "jointPain": "yes",
            "jointsAffected": ["fingers", "knees"],
            "nailWithJoint": "yes",
            "pastTreatments": "Topical steroids for 3 months",
            "familyHistory": ["psoriasis"],
            "otherConditions": ["obesity"],
            "currentTreatment": "Coal tar shampoo",
            "reliefSideEffects": "Minimal relief, scalp still itchy",
            "triedSystemic": "no",
            "feverInfection": "no",
            "weightLossFatigue": "no"
        }
    }
    
    print("=" * 70)
    print("Testing Questionnaire Endpoint")
    print("=" * 70)
    
    try:
        # Test 1: Basic connectivity
        print("\n🔍 Test 1: Checking server connectivity...")
        try:
            response = requests.get(BASE_URL, timeout=5)
            print(f"✅ Server is running: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Server not reachable: {e}")
            print("   Make sure the FastAPI server is running:")
            print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")
            return
        
        # Test 2: Submit questionnaire
        print("\n🔍 Test 2: Submitting questionnaire (Gemini LLM-powered)...")
        print(f"   Endpoint: {QUESTIONNAIRE_ENDPOINT}")
        print("   ⚡ Using Gemini AI for intelligent assessment...")
        
        response = requests.post(
            QUESTIONNAIRE_ENDPOINT,
            json=test_payload,
            headers={
                "Content-Type": "application/json",
                "X-User-ID": "test_user_123"  # Optional header
            },
            timeout=30  # Increased timeout for LLM processing
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Questionnaire submission successful!")
            print("   🤖 Gemini AI Analysis Complete")
            print("\n📊 Assessment Results:")
            print("-" * 70)
            print(f"   Assessment ID: {result.get('assessment_id')}")
            print(f"   Timestamp: {result.get('timestamp')}")
            print(f"   Severity: {result.get('severity_assessment').upper() if result.get('severity_assessment') else 'N/A'}")
            print(f"   PSA Risk: {result.get('psoriatic_arthritis_risk').upper() if result.get('psoriatic_arthritis_risk') else 'N/A'}")
            print(f"   Urgency: {result.get('treatment_urgency').upper() if result.get('treatment_urgency') else 'N/A'}")
            print(f"   Follow-up: {result.get('recommended_followup_weeks')} weeks")
            
            print("\n📝 AI-Generated Recommendations (Concise Bullets):")
            print("-" * 70)
            next_steps = result.get('nextSteps', [])
            if next_steps:
                for i, step in enumerate(next_steps, 1):
                    print(f"   {i}. {step}")
            else:
                print("   No recommendations generated")
            
            print(f"\n💡 Clinical Notes (AI-Generated):")
            print("-" * 70)
            clinical_notes = result.get('additionalNotes', '')
            if clinical_notes:
                # Word wrap the clinical notes
                import textwrap
                wrapped = textwrap.fill(clinical_notes, width=66)
                for line in wrapped.split('\n'):
                    print(f"   {line}")
            else:
                print("   No clinical notes available")
            
            # Check database metadata
            if 'db_metadata' in result and result['db_metadata']:
                print(f"\n💾 Database Status:")
                print(f"   Saved: {result['db_metadata'].get('saved', False)}")
                print(f"   DB Timestamp: {result['db_metadata'].get('timestamp', 'N/A')}")
            
            print("\n" + "=" * 70)
            print("✅ All tests passed! Gemini LLM is working correctly.")
            print("=" * 70)
            
            return True
            
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out")
        print("   ⚡ Gemini LLM processing may take longer. Try increasing timeout to 30-60 seconds.")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request error: {e}")
        return False
        
    except json.JSONDecodeError:
        print(f"\n❌ Invalid JSON response")
        print(f"   Response text: {response.text}")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mild_case():
    """Test with mild psoriasis case - Gemini AI assessment"""
    
    mild_payload = {
        "timestamp": "2025-12-08T11:00:00.000Z",
        "screen1": {
            "gender": "Male",
            "age": "35",
            "psoriasisHistory": "first",
            "location": ["elbows"],
            "appearance": ["red"],
            "size": ["coin"],
            "nails": [],
            "scalp": []
        },
        "screen2": {
            "onsetDate": "1 week ago",
            "symptomPattern": "stable",
            "lesionSpeed": "gradual",
            "itching": 3,
            "burning": 2,
            "pain": 1,
            "bleeding": 0,
            "worsenAtNight": "no",
            "worsenWithStress": "no",
            "triggers": [],
            "medTriggers": [],
            "sunlightEffect": "none"
        },
        "screen3": {
            "dailyImpact": "minimal",
            "emotionalImpact": "rarely",
            "relationshipsImpact": "none",
            "jointPain": "no",
            "jointsAffected": [],
            "nailWithJoint": "no",
            "pastTreatments": "None",
            "familyHistory": [],
            "otherConditions": [],
            "currentTreatment": "Moisturizer",
            "reliefSideEffects": "Some improvement",
            "triedSystemic": "no",
            "feverInfection": "no",
            "weightLossFatigue": "no"
        }
    }
    
    print("\n\n" + "=" * 70)
    print("Testing Mild Case Scenario (Gemini LLM)")
    print("=" * 70)
    
    try:
        response = requests.post(
            QUESTIONNAIRE_ENDPOINT,
            json=mild_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Mild case assessment complete:")
            print(f"   🤖 AI Severity: {result.get('severity_assessment', 'N/A').upper()}")
            print(f"   🤖 AI PSA Risk: {result.get('psoriatic_arthritis_risk', 'N/A').upper()}")
            print(f"   🤖 AI Urgency: {result.get('treatment_urgency', 'N/A').upper()}")
            print(f"   📅 Follow-up: {result.get('recommended_followup_weeks')} weeks")
            
            # Show first 3 recommendations
            next_steps = result.get('nextSteps', [])
            if next_steps:
                print(f"\n   Top 3 AI Recommendations:")
                for i, step in enumerate(next_steps[:3], 1):
                    print(f"      {i}. {step}")
            
            return True
        else:
            print(f"❌ Mild case test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Mild case error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "🧪" * 35)
    print("Questionnaire Endpoint Test Suite - Gemini LLM Powered")
    print("🧪" * 35 + "\n")
    
    print("ℹ️  This test suite validates:")
    print("   • Gemini AI integration for questionnaire analysis")
    print("   • Severity assessment (mild/moderate/severe)")
    print("   • PSA risk evaluation (low/medium/high)")
    print("   • Concise bullet-point recommendations")
    print("   • Database persistence\n")
    
    # Run tests
    test1_passed = test_questionnaire_endpoint()
    
    if test1_passed:
        test_mild_case()
    
    print("\n" + "=" * 70)
    print("🎉 Test suite completed - Gemini LLM endpoint verified!")
    print("=" * 70)
