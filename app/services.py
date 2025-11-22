import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

# Configuration
SNIPER_PATH = "models/sniper.pt"
JUDGE_PATH = "models/judge.pth"

class AIEngine:
    def __init__(self):
        self.sniper = None
        self.judge = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ AI Engine initializing on: {self.device}")

        # Transform pipeline for EfficientNet
        self.judge_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_models(self):
        """Loads YOLO and EfficientNet into memory."""
        if not os.path.exists(SNIPER_PATH) or not os.path.exists(JUDGE_PATH):
            print("❌ CRITICAL: Models not found in /models folder!")
            return

        print("⏳ Loading Sniper...")
        self.sniper = YOLO(SNIPER_PATH)

        print("⏳ Loading Judge...")
        try:
            self.judge = models.efficientnet_b0(weights=None)
            self.judge.classifier[1] = nn.Linear(1280, 3) # Mild, Mod, Severe
            
            state_dict = torch.load(JUDGE_PATH, map_location=self.device)
            self.judge.load_state_dict(state_dict)
            self.judge.to(self.device)
            self.judge.eval()
            print("✅ Models Loaded Successfully.")
        except Exception as e:
            print(f"❌ Error loading Judge: {e}")

    def calculate_lesion_metrics(self, pil_crop):
        """
        Hybrid Grading: AI Baseline + OpenCV Tweaks
        Returns dictionary with E, I, D scores (0-4) and Global Score (0-10).
        """
        # 1. AI BASELINE (General Severity 0-4)
        img_tensor = self.judge_transform(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.judge(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            p_mild, p_mod, p_sev = probs[0]
            
            # Weighted Average: Mild=1, Mod=2.5, Sev=4
            ai_base = (p_mild * 1.0) + (p_mod * 2.5) + (p_sev * 4.0)

        # 2. OPENCV TWEAKS
        # Convert PIL (RGB) to OpenCV (BGR)
        cv_img = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        
        # --- A. Erythema (Redness) ---
        b, g, r = cv2.split(cv_img)
        avg_red, avg_green = np.mean(r), np.mean(g)
        
        e_score = ai_base.item()
        if avg_red > (avg_green * 1.3): e_score += 1.0    # Deep Red
        elif avg_red > (avg_green * 1.1): e_score += 0.5  # Pink

        # --- B. Desquamation (Scaling) ---
        # Convert to HSV to find white/desaturated areas
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1] # Saturation channel
        
        # Calculate ratio of low-saturation pixels (White/Silver scale)
        white_ratio = np.sum(s < 50) / s.size
        
        d_score = ai_base.item()
        if white_ratio > 0.30: d_score += 1.0   # Heavy Scale
        elif white_ratio > 0.10: d_score += 0.5 # Moderate Scale
        
        # --- C. Induration (Thickness) ---
        # Relies mostly on AI shadow detection
        i_score = ai_base.item()

        # 3. FINAL CLAMP & FORMAT
        final_e = int(round(min(max(e_score, 0), 4)))
        final_i = int(round(min(max(i_score, 0), 4)))
        final_d = int(round(min(max(d_score, 0), 4)))
        
        # Convert 0-4 scale to 0-10 scale for the Global Score
        # (Sum of symptoms / Max possible sum) * 10
        global_0_10 = ((final_e + final_i + final_d) / 12.0) * 10.0
        
        return {
            "erythema": final_e,
            "induration": final_i,
            "desquamation": final_d,
            "severity_score": round(global_0_10, 2)
        }

    def analyze_image(self, original_image: Image.Image):
        """Main Pipeline: Detect -> Crop -> Grade -> Average"""
        
        # Fix EXIF Rotation (Crucial for mobile uploads)
        original_image = ImageOps.exif_transpose(original_image)

        # 1. Run Sniper
        results = self.sniper(original_image, verbose=False)
        result = results[0]

        # Handle Clear Skin
        if not result.masks:
            return {
                "diagnosis": "Clear",
                "global_score": 0.0,
                "lesions_found": 0,
                "details": []
            }

        total_area = 0
        weighted_score_sum = 0
        lesions_data = []

        # 2. Loop Lesions
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())
            
            # Sanity Check Crop
            w, h = original_image.size
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            area = (x2 - x1) * (y2 - y1)
            
            if area < 50: continue # Ignore noise

            # Crop & Grade
            crop = original_image.crop((x1, y1, x2, y2))
            metrics = self.calculate_lesion_metrics(crop)
            
            score = metrics["severity_score"]
            
            # Add to weighted sum
            weighted_score_sum += (score * area)
            total_area += area
            
            # Local Diagnosis
            local_diag = "Mild" if score < 3 else "Moderate" if score < 7 else "Severe"

            lesions_data.append({
                "id": i + 1,
                "diagnosis": local_diag,
                "severity_score": score,
                "area_pixels": int(area),
                "erythema": metrics["erythema"],
                "induration": metrics["induration"],
                "desquamation": metrics["desquamation"]
            })

        # 3. Final Aggregate
        if total_area > 0:
            global_score = weighted_score_sum / total_area
        else:
            global_score = 0

        if global_score < 3.0: status = "Mild"
        elif global_score < 7.0: status = "Moderate"
        else: status = "Severe"

        return {
            "diagnosis": status,
            "global_score": round(global_score, 2),
            "lesions_found": len(lesions_data),
            "details": lesions_data
        }

# Singleton
ai_engine = AIEngine()