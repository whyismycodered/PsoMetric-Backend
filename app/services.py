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
        """Converts a numpy/OpenCV image to a Base64 string for the Frontend."""
        rgb_img = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=70) 
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def white_balance(self, cv_img):
        """
        Advanced 'Gray World' White Balance.
        """
        result = cv_img.copy()
        b, g, r = cv2.split(result)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        
        # Handle pitch black images
        if b_mean == 0 or g_mean == 0 or r_mean == 0: return cv_img

        k = (b_mean + g_mean + r_mean) / 3
        b = np.clip(b * (k / b_mean), 0, 255).astype(np.uint8)
        g = np.clip(g * (k / g_mean), 0, 255).astype(np.uint8)
        r = np.clip(r * (k / r_mean), 0, 255).astype(np.uint8)
        return cv2.merge((b, g, r))

    def calculate_lesion_metrics(self, pil_crop, lesion_mask=None):
        """
        Advanced Scoring: Uses CIELAB Color Space & Relative Skin Tone Comparison.
        """
        # 1. AI BASELINE (The "Gut Feeling")
        img_tensor = self.judge_transform(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.judge(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            p_mild, p_mod, p_sev = probs[0]
            
            # Weighted Average
            ai_score = (p_mild * 1.0) + (p_mod * 2.5) + (p_sev * 4.0)

        # 2. COMPUTER VISION (The "Pixel Facts")
        cv_img = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        cv_img = self.white_balance(cv_img)
        
        # Handle Mask
        if lesion_mask is not None:
            lesion_mask = cv2.resize(lesion_mask, (cv_img.shape[1], cv_img.shape[0]))
            _, lesion_mask = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            lesion_mask = np.ones((cv_img.shape[0], cv_img.shape[1]), dtype=np.uint8) * 255

        # Create 'Background Mask' (Healthy Skin) by inverting lesion mask
        bg_mask = cv2.bitwise_not(lesion_mask)

        # --- A. ERYTHEMA (Using CIELAB A-Channel) ---
        # Convert to LAB color space (L=Lightness, A=Green-Red, B=Blue-Yellow)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Calculate Mean 'A' (Redness) inside lesion vs outside
        lesion_redness = cv2.mean(a_channel, mask=lesion_mask)[0]
        skin_redness = cv2.mean(a_channel, mask=bg_mask)[0]

        # If no background found (crop is 100% lesion), assume standard skin A=128
        if skin_redness == 0: skin_redness = 128.0

        # Delta-A: How much redder is the lesion than the skin?
        # A value of ~128 is neutral. >128 is red.
        delta_a = lesion_redness - skin_redness
        
        # Scoring Logic (calibrated values)
        if delta_a > 25: e_cv = 4
        elif delta_a > 18: e_cv = 3
        elif delta_a > 10: e_cv = 2
        elif delta_a > 4: e_cv = 1
        else: e_cv = 0

        # --- B. DESQUAMATION (HSV + Texture Energy) ---
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        
        # White Scale = Low Saturation inside lesion
        white_pixels = np.sum((s < 40) & (lesion_mask > 0))
        total_pixels = np.count_nonzero(lesion_mask) + 1e-5
        white_ratio = white_pixels / total_pixels

        if white_ratio > 0.35: d_cv = 4
        elif white_ratio > 0.20: d_cv = 3
        elif white_ratio > 0.10: d_cv = 2
        elif white_ratio > 0.05: d_cv = 1
        else: d_cv = 0

        # --- C. INDURATION (Gradient Magnitude) ---
        # Thickness creates sharp edges and shadows. 
        # We calculate Laplacian Variance (Texture Roughness) inside the lesion.
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to gray image
        masked_gray = cv2.bitwise_and(gray, gray, mask=lesion_mask)
        
        # Calculate gradients (roughness/edges)
        laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
        roughness = laplacian.var()
        
        # Roughness map: Smooth < 100, Very Rough > 1000
        if roughness > 1200: i_cv = 4
        elif roughness > 800: i_cv = 3
        elif roughness > 400: i_cv = 2
        elif roughness > 100: i_cv = 1
        else: i_cv = 0
        
        # Induration Logic Sanity Check:
        # If Desquamation (Scale) is high, Induration is usually high (Thick plaque)
        if d_cv >= 3: 
            i_cv = max(i_cv, 3)
        # Induration cannot be 2 points higher than Erythema (Can't be thick but not red)
        if i_cv > (e_cv + 1):
            i_cv = e_cv + 1

        # 3. FUSION (The Blend)
        # We trust CV metrics (Objecitve) more than AI (Subjective) for sub-scores
        # But we blend them to be safe.
        
        # Weighted Blend: 70% Computer Vision, 30% AI
        final_e = int(round((e_cv * 0.7) + (ai_score.item() * 0.3)))
        final_d = int(round((d_cv * 0.7) + (ai_score.item() * 0.3)))
        final_i = int(round((i_cv * 0.7) + (ai_score.item() * 0.3)))
        
        # Floor/Ceiling
        final_e = min(max(final_e, 0), 4)
        final_d = min(max(final_d, 0), 4)
        final_i = min(max(final_i, 0), 4)

        # 4. GLOBAL SCORE
        # PASI Formula weighting (Erythema matters slightly more for severity perception)
        global_0_10 = ((final_e + final_i + final_d) / 12.0) * 10.0
        
        return {
            "erythema": final_e,
            "induration": final_i,
            "desquamation": final_d,
            "severity_score": round(global_0_10, 2)
        }

    def analyze_image(self, original_image: Image.Image):
        """Main Pipeline: Detect -> Crop -> Grade -> Average"""
        
        # Fix EXIF Rotation
        original_image = ImageOps.exif_transpose(original_image)

        # 1. Run Sniper (Segmentation)
        # conf=0.10 ensures we catch even faint mild lesions
        results = self.sniper(original_image, verbose=False, conf=0.10)
        result = results[0]
        
        # --- VISUALIZATION ---
        annotated_numpy = result.plot() 
        b64_string = self.image_to_base64(annotated_numpy)
        # ---------------------

        # Handle Clear Skin
        if not result.masks:
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

        # Prepare Full Masks (Resize to match original image)
        # YOLO returns masks in a smaller size, we need to scale them up
        if result.masks:
            full_masks = torch.nn.functional.interpolate(
                result.masks.data.unsqueeze(1), 
                size=original_image.size[::-1], # (H, W)
                mode="bilinear", 
                align_corners=False
            ).squeeze(1).cpu().numpy()
        else:
            full_masks = None

        # 2. Loop Lesions
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())
            
            # Sanity Check Crops
            w, h = original_image.size
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            area = (x2 - x1) * (y2 - y1)
            
            if area < 50: continue # Ignore tiny noise

            # Crop Image
            crop = original_image.crop((x1, y1, x2, y2))
            
            # Crop Mask (The Exact Shape)
            if full_masks is not None:
                mask_crop = full_masks[i][y1:y2, x1:x2]
                mask_crop = (mask_crop * 255).astype(np.uint8)
            else:
                mask_crop = None

            # Grade with ROI Masking
            metrics = self.calculate_lesion_metrics(crop, lesion_mask=mask_crop)
            
            score = metrics["severity_score"]
            
            # Weighted Sum
            weighted_score_sum += (score * area)
            total_area += area
            
            # Local Diagnosis
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

        # 3. Final Aggregate
        if total_area > 0:
            global_score = weighted_score_sum / total_area
        else:
            global_score = 0

        # Diagnosis Logic
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