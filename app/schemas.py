from pydantic import BaseModel
from typing import List, Optional

# ==================== IMAGE ANALYSIS SCHEMAS ====================

# Detail for a single lesion found in the image
class LesionDetail(BaseModel):
    id: int
    diagnosis: str          # Mild, Moderate, Severe
    severity_score: float   # Global score 0-10 (for sorting)
    area_pixels: int
    
    # --- PASI METRICS (0-4 Scale) ---
    erythema: int           # Redness
    induration: int         # Thickness
    desquamation: int       # Scaling

# Database metadata
class DatabaseMetadata(BaseModel):
    analysis_id: str
    timestamp: str
    saved: bool

# The Main Response Object
class PsoriasisAnalysisResponse(BaseModel):
    diagnosis: str          # Global Diagnosis
    global_score: float     # Weighted Average 0-10
    lesions_found: int
    annotated_image_base64: Optional[str] = None # The "Heatmap" Image
    details: List[LesionDetail]
    db_metadata: Optional[DatabaseMetadata] = None  # Database save info
    error: Optional[str] = None # Error message if validation fails


# ==================== QUESTIONNAIRE SCHEMAS ====================

class QuestionnaireRequest(BaseModel):
    # Basic Info
    gender: str
    age: str
    psoriasisHistory: str
    
    # Symptoms
    location: List[str]
    appearance: List[str]
    size: List[str]
    
    # Severity (0-10 scale)
    itching: int
    pain: int
    
    # Impact & Joints
    dailyImpact: str
    jointPain: str
    jointsAffected: List[str]
    
    # Treatment
    currentTreatment: str

class QuestionnaireResponse(BaseModel):
    assessment_id: str
    timestamp: str
    severity_assessment: str
    psoriatic_arthritis_risk: str
    nextSteps: List[str]
    additionalNotes: str
    treatment_urgency: str
    recommended_followup_weeks: int
    db_metadata: Optional[DatabaseMetadata] = None  # Database save info


