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
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # Calculate redness metrics within lesion area
        avg_red = cv2.mean(r, mask=lesion_mask)[0]
        avg_green = cv2.mean(g, mask=lesion_mask)[0]
        avg_blue = cv2.mean(b, mask=lesion_mask)[0]
        avg_saturation = cv2.mean(hsv[:, :, 1], mask=lesion_mask)[0]
        
        # Red dominance ratio (how much redder than green/blue)
        red_ratio = avg_red / (avg_green + 1e-5)
        red_purity = avg_red / (avg_red + avg_green + avg_blue + 1e-5)
        
        # Start with AI baseline
        e_score = ai_base.item()
        
        # Multi-factor redness assessment
        if red_ratio < 1.05 or red_purity < 0.35:
            # Very minimal redness - likely clear or very mild
            e_score = min(e_score, 1.0)
        elif red_ratio < 1.15 or red_purity < 0.38:
            # Mild redness with low saturation
            e_score = min(e_score, 2.0)
        elif red_ratio > 1.4 and avg_saturation > 80:
            # Intense, saturated redness - upgrade severity
            e_score = max(e_score, 3.0)
            if red_ratio > 1.6:
                e_score += 0.5
        elif red_ratio > 1.3:
            # Moderate-high redness
            e_score += 0.3
        
        # Prevent unrealistic scores
        e_score = max(1.0, min(4.0, e_score))

        # --- B. Desquamation (Scaling) ---
        # Enhanced texture analysis to distinguish scale from glare
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        s = hsv[:, :, 1]  # Saturation
        v = hsv[:, :, 2]  # Value (Brightness)
        
        # 1. Identify candidate white/light pixels (potential scale or glare)
        # Stricter saturation threshold to avoid colored areas
        white_candidates = (s < 35) & (v > 90) & (lesion_mask > 0)
        
        # 2. Multi-scale texture analysis for roughness detection
        # Laplacian for edge detection (high frequency texture)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.uint8(np.absolute(laplacian))
        
        # Variance-based texture (local roughness)
        kernel_size = 5
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        variance = cv2.absdiff(gray, blur)
        
        # Combine texture indicators: both edge detection and local variance
        rough_mask = (laplacian_abs > 15) | (variance > 12)
        
        # 3. Scale pixels must be white AND have rough texture
        confirmed_scale_pixels = np.sum(white_candidates & rough_mask)
        total_lesion_pixels = np.count_nonzero(lesion_mask) + 1e-5
        
        scale_ratio = confirmed_scale_pixels / total_lesion_pixels
        
        # Also check average texture intensity in scale areas
        if confirmed_scale_pixels > 0:
            scale_texture_intensity = cv2.mean(laplacian_abs, 
                                               mask=np.uint8(white_candidates & rough_mask))[0]
        else:
            scale_texture_intensity = 0
        
        # Start with AI baseline
        d_score = ai_base.item()
        
        # Visual veto and enhancement based on scale coverage and texture intensity
        if scale_ratio < 0.015:
            # Minimal to no visible scaling
            d_score = min(d_score, 1.0)
        elif scale_ratio < 0.035:
            # Very light scaling
            d_score = min(d_score, 1.5)
            if scale_texture_intensity > 25:
                d_score += 0.3  # Boost if texture is pronounced
        elif scale_ratio < 0.08:
            # Mild to moderate scaling
            d_score = min(d_score, 2.5)
            if scale_texture_intensity > 30:
                d_score += 0.5
        elif scale_ratio > 0.18:
            # Heavy scaling coverage
            d_score = max(d_score, 3.0)
            if scale_texture_intensity > 35:
                d_score += 0.8
        elif scale_ratio > 0.08:
            # Moderate scaling
            d_score += 0.5
            if scale_texture_intensity > 32:
                d_score += 0.3
        
        # Ensure reasonable bounds
        d_score = max(0.0, min(4.0, d_score))

        # --- C. Induration (Thickness) ---
        # Induration is harder to measure from 2D images, but we can use:
        # 1. Shadow/depth cues (darker borders, brightness variation)
        # 2. Texture density (raised areas often have more pronounced texture)
        # 3. Color intensity (thicker lesions often appear darker/more saturated)
        
        # Calculate brightness variation (std dev) - raised areas cast micro-shadows
        lesion_pixels = gray[lesion_mask > 0]
        if len(lesion_pixels) > 0:
            brightness_std = np.std(lesion_pixels)
        else:
            brightness_std = 0
        
        # Check for darker borders (elevation indicator)
        # Erode mask to get inner region, compare with border
        kernel = np.ones((5,5), np.uint8)
        inner_mask = cv2.erode(lesion_mask, kernel, iterations=1)
        border_mask = cv2.subtract(lesion_mask, inner_mask)
        
        if np.count_nonzero(border_mask) > 0 and np.count_nonzero(inner_mask) > 0:
            avg_border_brightness = cv2.mean(gray, mask=border_mask)[0]
            avg_inner_brightness = cv2.mean(gray, mask=inner_mask)[0]
            border_darkness = avg_inner_brightness - avg_border_brightness
        else:
            border_darkness = 0
        
        # Texture density within lesion
        texture_density = cv2.mean(laplacian_abs, mask=lesion_mask)[0]
        
        # Color intensity (saturation + value)
        lesion_saturation = cv2.mean(hsv[:, :, 1], mask=lesion_mask)[0]
        lesion_brightness = cv2.mean(hsv[:, :, 2], mask=lesion_mask)[0]
        color_intensity = (lesion_saturation / 255.0) * (lesion_brightness / 255.0)
        
        # Start with AI baseline
        i_score = ai_base.item()
        
        # Induration indicators analysis
        induration_indicators = 0
        
        # High brightness variation suggests elevation/depth
        if brightness_std > 25:
            induration_indicators += 1
            if brightness_std > 40:
                i_score += 0.4
        
        # Darker borders suggest raised edges
        if border_darkness > 8:
            induration_indicators += 1
            if border_darkness > 15:
                i_score += 0.5
        
        # Dense texture suggests thickness
        if texture_density > 18:
            induration_indicators += 1
            if texture_density > 28:
                i_score += 0.3
        
        # High color intensity can indicate thickness
        if color_intensity > 0.4:
            if lesion_saturation > 100:
                i_score += 0.2
        
        # Multi-indicator boost
        if induration_indicators >= 3:
            i_score += 0.5
        elif induration_indicators >= 2:
            i_score += 0.3
        elif induration_indicators == 0:
            # No clear thickness indicators - likely flat
            i_score = min(i_score, 2.0)
        
        # Logical constraint: Induration shouldn't vastly exceed erythema
        # (You can't have very thick lesions with no redness)
        if i_score > (e_score + 1.2):
            i_score = e_score + 1.2
        
        # Ensure reasonable bounds
        i_score = max(1.0, min(4.0, i_score))

        # 3. FINAL SCORING
        # Round to nearest 0.5 for more granular scoring, then convert to integers
        final_e = max(1, min(4, int(round(e_score * 2) / 2)))  # Floor 1, range 1-4
        final_i = max(1, min(4, int(round(i_score * 2) / 2)))  # Floor 1, range 1-4
        final_d = max(0, min(4, int(round(d_score * 2) / 2)))  # Floor 0, range 0-4
        
        # Global severity score (0-10 scale)
        # Max possible: (4+4+4)/12 * 10 = 10.0
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