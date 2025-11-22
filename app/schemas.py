from pydantic import BaseModel
from typing import List, Optional

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

# The Main Response Object
class PsoriasisAnalysisResponse(BaseModel):
    diagnosis: str          # Global Diagnosis
    global_score: float     # Weighted Average 0-10
    lesions_found: int
    annotated_image_base64: Optional[str] = None # The "Heatmap" Image
    details: List[LesionDetail]