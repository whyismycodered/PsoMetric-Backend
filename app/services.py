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

# --- CONFIGURATION ---
SNIPER_PATH = "models/sniper.pt"
JUDGE_PATH = "models/judge.pth"

class AIEngine:
    def __init__(self):
        self.sniper = None
        self.judge = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ AI Engine initializing on: {self.device}")

        # Transform pipeline for EfficientNet (The Judge)
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

        print("⏳ Loading Sniper (Segmentation)...")
        self.sniper = YOLO(SNIPER_PATH)

        print("⏳ Loading Judge (Classifier)...")
        try:
            self.judge = models.efficientnet_b0(weights=None)
            self.judge.classifier[1] = nn.Linear(1280, 3) # Mild, Mod, Severe
            
            state_dict = torch.load(JUDGE_PATH, map_location=self.device)
            self.judge.load_state_dict(state_dict)
            self.judge.to(self.device)
            self.judge.eval()
            print("✅ AI Models Loaded Successfully.")
        except Exception as e:
            print(f"❌ Error loading Judge: {e}")

    def image_to_base64(self, numpy_image):
        """Converts a numpy/OpenCV image to a Base64 string."""
        rgb_img = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=70) 
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def white_balance(self, cv_img):
        """
        Corrects yellow indoor lighting ("Gray World" assumption).
        """
        result = cv_img.copy()
        b, g, r = cv2.split(result)
        
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        if b_mean == 0 or g_mean == 0 or r_mean == 0: return cv_img

        k = (b_mean + g_mean + r_mean) / 3
        
        b = np.clip(b * (k / b_mean), 0, 255).astype(np.uint8)
        g = np.clip(g * (k / g_mean), 0, 255).astype(np.uint8)
        r = np.clip(r * (k / r_mean), 0, 255).astype(np.uint8)
        
        return cv2.merge((b, g, r))

    def calculate_lesion_metrics(self, pil_crop, lesion_mask=None):
        """
        Hybrid Grading with ROI Masking, Glare Filtering, and Visual Veto.
        """
        # 1. AI BASELINE
        img_tensor = self.judge_transform(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.judge(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Weighted Average (Conservative)
            # Mild=1, Mod=2.2, Sev=3.8
            p_mild, p_mod, p_sev = probs[0]
            ai_base = (p_mild * 1.0) + (p_mod * 2.2) + (p_sev * 3.8)

        # 2. OPENCV ANALYSIS
        cv_img = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        cv_img = self.white_balance(cv_img)
        
        # Prepare Mask
        if lesion_mask is not None:
            lesion_mask = cv2.resize(lesion_mask, (cv_img.shape[1], cv_img.shape[0]))
            _, lesion_mask = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            lesion_mask = np.ones((cv_img.shape[0], cv_img.shape[1]), dtype=np.uint8) * 255

        # --- A. Erythema (Redness) ---
        b, g, r = cv2.split(cv_img)
        avg_red = cv2.mean(r, mask=lesion_mask)[0]
        avg_green = cv2.mean(g, mask=lesion_mask)[0]
        
        red_ratio = avg_red / (avg_green + 1e-5)
        e_score = ai_base.item()
        
        # Visual Veto: Redness
        if red_ratio < 1.1: e_score = min(e_score, 1.0)
        elif red_ratio < 1.25: e_score = min(e_score, 2.0)
        elif red_ratio > 1.5: e_score += 0.5

        # --- B. Desquamation (Scaling) ---
        # FIX: Distinguish Scale (Rough) from Glare (Smooth)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1] # Saturation
        v = hsv[:, :, 2] # Value (Brightness)
        
        # 1. Find Candidate White Pixels (Low Saturation + High Brightness)
        # We increase strictness: Saturation must be VERY low (< 30)
        white_candidates = (s < 30) & (v > 100) & (lesion_mask > 0)
        
        # 2. TEXTURE CHECK (The Glare Filter)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # "Roughness Mask": Pixels where edges are strong
        rough_mask = laplacian > 20
        
        # 3. COMBINE: It must be WHITE AND ROUGH
        confirmed_scale_pixels = np.sum(white_candidates & rough_mask)
        total_pixels = np.count_nonzero(lesion_mask) + 1e-5
        
        scale_ratio = confirmed_scale_pixels / total_pixels
        d_score = ai_base.item()
        
        # Visual Veto: Scaling
        if scale_ratio < 0.02: 
            d_score = min(d_score, 1.0) # Force Mild/None
        elif scale_ratio > 0.15: 
            d_score += 1.0 # Heavy scale
        elif scale_ratio > 0.05:
            d_score += 0.5 # Moderate scale

        # --- C. Induration (Thickness) ---
        i_score = ai_base.item()
        if i_score > (e_score + 1.0): i_score = e_score + 1.0

        # 3. FINAL SCORING
        final_e = int(round(min(max(e_score, 1), 4))) # Floor 1
        final_i = int(round(min(max(i_score, 1), 4))) # Floor 1
        final_d = int(round(min(max(d_score, 0), 4))) # Floor 0 (Scale can be 0)
        
        global_0_10 = ((final_e + final_i + final_d) / 12.0) * 10.0
        
        return {
            "erythema": final_e,
            "induration": final_i,
            "desquamation": final_d,
            "severity_score": round(global_0_10, 2)
        }

    def analyze_image(self, original_image: Image.Image):
        """Main Pipeline: Detect -> Crop -> Grade -> Average"""
        original_image = ImageOps.exif_transpose(original_image)

        # 1. Run Sniper (Low Threshold for Mild Cases)
        results = self.sniper(original_image, verbose=False, conf=0.10)
        result = results[0]
        
        # Visualization
        annotated_numpy = result.plot() 
        b64_string = self.image_to_base64(annotated_numpy)

        # --- 🛡️ FALLBACK: If Sniper misses, force Center Crop ---
        if not result.masks:
            # Crop Center 75%
            w, h = original_image.size
            crop = original_image.crop((w*0.125, h*0.125, w*0.875, h*0.875))
            
            # Force Grade
            metrics = self.calculate_lesion_metrics(crop, lesion_mask=None)
            score = metrics["severity_score"]
            
            # If score is extremely low, accept Clear. Otherwise, report findings.
            if score < 0.5:
                return {
                    "diagnosis": "Clear",
                    "global_score": 0.0, 
                    "lesions_found": 0, 
                    "annotated_image_base64": b64_string, 
                    "details": []
                }
            
            # Return "Assumed" Lesion
            local_diag = "Mild" if score < 4.0 else "Moderate" if score < 7.5 else "Severe"
            return {
                "diagnosis": local_diag,
                "global_score": score,
                "lesions_found": 1, 
                "annotated_image_base64": b64_string, 
                "details": [{
                    "id": 1,
                    "diagnosis": local_diag,
                    "severity_score": score,
                    "area_pixels": int((w*0.75)*(h*0.75)),
                    "erythema": metrics["erythema"],
                    "induration": metrics["induration"],
                    "desquamation": metrics["desquamation"]
                }]
            }

        # --- STANDARD LOGIC ---
        total_area = 0
        weighted_score_sum = 0
        lesions_data = []

        if result.masks:
            full_masks = torch.nn.functional.interpolate(
                result.masks.data.unsqueeze(1), 
                size=original_image.size[::-1],
                mode="bilinear", 
                align_corners=False
            ).squeeze(1).cpu().numpy()
        else:
            full_masks = None

        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())
            w, h = original_image.size
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            area = (x2 - x1) * (y2 - y1)
            if area < 20: continue 

            crop = original_image.crop((x1, y1, x2, y2))
            
            if full_masks is not None:
                mask_crop = full_masks[i][y1:y2, x1:x2]
                mask_crop = (mask_crop * 255).astype(np.uint8)
            else:
                mask_crop = None

            metrics = self.calculate_lesion_metrics(crop, lesion_mask=mask_crop)
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

# Singleton Instance
ai_engine = AIEngine()