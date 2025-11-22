from pydantic import BaseModel
from typing import List, Optional

# Detail for a single lesion found in the image
class LesionDetail(BaseModel):
    id: int
    diagnosis: str          # Mild/Mod/Severe
    severity_score: float   # The 0-10 Global Score (for sorting/color coding)
    area_pixels: int
    
    # --- NEW MEDICAL METRICS (PASI Sub-scores 0-4) ---
    erythema: int           # Redness
    induration: int         # Thickness
    desquamation: int       # Scaling

# The Main Response Object
class PsoriasisAnalysisResponse(BaseModel):
    diagnosis: str
    global_score: float
    lesions_found: int
    processed_image_url: Optional[str] = None
    details: List[LesionDetail]