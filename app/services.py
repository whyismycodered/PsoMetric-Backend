import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import base64
import io

# Configuration
SNIPER_PATH = "models/sniper.pt"
JUDGE_PATH = "models/judge.pth"

class AIEngine:
    def __init__(self):
        self.sniper = None
        self.judge = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ AI Engine initializing on: {self.device}")

        self.judge_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_models(self):
        if not os.path.exists(SNIPER_PATH) or not os.path.exists(JUDGE_PATH):
            print("❌ CRITICAL: Models not found!")
            return

        print("⏳ Loading Sniper...")
        self.sniper = YOLO(SNIPER_PATH)

        print("⏳ Loading Judge...")
        try:
            self.judge = models.efficientnet_b0(weights=None)
            self.judge.classifier[1] = nn.Linear(1280, 3) 
            
            state_dict = torch.load(JUDGE_PATH, map_location=self.device)
            self.judge.load_state_dict(state_dict)
            self.judge.to(self.device)
            self.judge.eval()
            print("✅ Models Loaded Successfully.")
        except Exception as e:
            print(f"❌ Error loading Judge: {e}")

    def image_to_base64(self, numpy_image):
        rgb_img = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=70) 
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def white_balance(self, cv_img):
        result = cv_img.copy()
        b, g, r = cv2.split(result)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        
        if b_mean == 0 or g_mean == 0 or r_mean == 0: return cv_img

        k = (b_mean + g_mean + r_mean) / 3
        b = np.clip(b * (k / b_mean), 0, 255).astype(np.uint8)
        g = np.clip(g * (k / g_mean), 0, 255).astype(np.uint8)
        r = np.clip(r * (k / r_mean), 0, 255).astype(np.uint8)
        return cv2.merge((b, g, r))

    def calculate_lesion_metrics(self, pil_crop):
        """Hybrid Grading with Minimum Floor"""
        
        # 1. AI BASELINE
        img_tensor = self.judge_transform(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.judge(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # WINNER TAKES ALL STRATEGY
            dominant_class = torch.argmax(probs).item()

            if dominant_class == 0:   # Mild
                ai_base = 1.0
                is_mild_anchor = True
            elif dominant_class == 1: # Mod
                ai_base = 2.5
                is_mild_anchor = False
            else:                     # Sev
                ai_base = 4.0
                is_mild_anchor = False

        # 2. OPENCV TWEAKS
        cv_img = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        cv_img = self.white_balance(cv_img)
        
        # Erythema
        b, g, r = cv2.split(cv_img)
        avg_red, avg_green = np.mean(r), np.mean(g)
        e_score = ai_base
        
        if is_mild_anchor:
            if avg_red > (avg_green * 1.6): e_score += 1.0
        else:
            if avg_red > (avg_green * 1.4): e_score += 1.0
            elif avg_red > (avg_green * 1.2): e_score += 0.5
        
        # Scaling
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1] 
        white_ratio = np.sum(s < 40) / s.size
        d_score = ai_base
        
        if is_mild_anchor:
            if white_ratio > 0.35: d_score += 1.0
        else:
            if white_ratio > 0.25: d_score += 1.0
            elif white_ratio > 0.10: d_score += 0.5

        i_score = ai_base # Induration relies on AI base

        # 3. SAFEGUARD: Ensure scores don't drop to 0 if AI saw something
        # If AI detected a lesion, the minimum PASI score for any symptom is usually 1
        # unless it's completely absent. But for "Mild Psoriasis", Induration/Redness 
        # is rarely 0.
        
        final_e = int(round(min(max(e_score, 1), 4))) # Floor of 1
        final_i = int(round(min(max(i_score, 1), 4))) # Floor of 1
        final_d = int(round(min(max(d_score, 0), 4))) # Scale can be 0 (smooth)
        
        # Global Score Calculation
        global_0_10 = ((final_e + final_i + final_d) / 12.0) * 10.0
        
        return {
            "erythema": final_e,
            "induration": final_i,
            "desquamation": final_d,
            "severity_score": round(global_0_10, 2)
        }

    def analyze_image(self, original_image: Image.Image):
        original_image = ImageOps.exif_transpose(original_image)

        # 🔍 FIX 1: LOWER CONFIDENCE THRESHOLD
        # conf=0.10 means "If you are even 10% sure it's a lesion, detect it."
        # This is crucial for Mild psoriasis which is faint.
        results = self.sniper(original_image, verbose=False, conf=0.10)
        result = results[0]
        
        annotated_numpy = result.plot() 
        b64_string = self.image_to_base64(annotated_numpy)

        if not result.masks:
            # If we STILL find nothing, it really is clear skin.
            return {
                "diagnosis": "Clear",
                "global_score": 0.0,
                "lesions_found": 0,
                "annotated_image_base64": b64_string,
                "details": []
            }

        total_area = 0
        weighted_score_sum = 0
        lesions_data = []

        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())
            w, h = original_image.size
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            # Relax area filter for mild spots (allow smaller spots)
            area = (x2 - x1) * (y2 - y1)
            if area < 20: continue 

            crop = original_image.crop((x1, y1, x2, y2))
            metrics = self.calculate_lesion_metrics(crop)
            score = metrics["severity_score"]
            
            weighted_score_sum += (score * area)
            total_area += area
            
            local_diag = "Mild" if score < 4.0 else "Moderate" if score < 7.5 else "Severe"

            lesions_data.append({
                "id": i + 1,
                "diagnosis": local_diag,
                "severity_score": score,
                "area_pixels": int(area),
                "erythema": metrics["erythema"],
                "induration": metrics["induration"],
                "desquamation": metrics["desquamation"]
            })

        if total_area > 0:
            global_score = weighted_score_sum / total_area
        else:
            global_score = 0

        # Widen Mild Range
        if global_score < 4.0: status = "Mild"
        elif global_score < 7.5: status = "Moderate"
        else: status = "Severe"

        return {
            "diagnosis": status,
            "global_score": round(global_score, 2),
            "lesions_found": len(lesions_data),
            "annotated_image_base64": b64_string,
            "details": lesions_data
        }

ai_engine = AIEngine()