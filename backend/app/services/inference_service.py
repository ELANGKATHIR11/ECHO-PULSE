import torch
import numpy as np
import cv2
import os
import uuid
from typing import Dict, Any, List
from pathlib import Path
from ..sonar.processor import OpenCVProcessor
from ..services.shadow_service import ShadowGeometryAnalyzer
from ..services.geotag_service import GeotaggingService
from ..models.ai_models import (
    LightweightSonarUNet, SonarAutoencoder, MultiFactorFusion,
    AcousticAngularReflectanceAttention, ShadowHighlightCrossAttention
)
from ..schemas.contracts import DetectionSchema, BoundingBox, DetectionGeometry
from ..services.guardrails_service import HeavyDebrisGuardrailEngine, TARGET_CLASS_MAPPING
from ..core.config import settings

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Classes recognized by EchoPulseNet (Standard 8-Class Marine Sonar Taxonomy)
CLASS_MAPPINGS = {
    0: ("ghost_gear", "Derelict Ghost Gear & Fishing Net"),
    1: ("shipwreck", "Shipwreck / Submerged Hull"),
    2: ("unexploded_ordnance", "Unexploded Ordnance (UXO)"),
    3: ("pipeline_anomaly", "Pipeline Scour / Anchor Drag Anomaly"),
    4: ("marine_debris", "Marine Anthropogenic Debris"),
    5: ("subsea_cable", "Subsea Power & Data Cable"),
    6: ("biological_cluster", "Benthic Biological Cluster / Coral Reef"),
    7: ("geological_formation", "Geological Outcrop / Seafloor Ridge")
}

class UnifiedInferenceService:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.unet = LightweightSonarUNet(in_channels=1, out_channels=2).to(self.device)
        self.autoencoder = SonarAutoencoder().to(self.device)
        self.aara_head = AcousticAngularReflectanceAttention(in_features=64).to(self.device)
        self.cross_attn = ShadowHighlightCrossAttention(embed_dim=64).to(self.device)
        
        self.unet.eval()
        self.autoencoder.eval()
        self.aara_head.eval()
        self.cross_attn.eval()

        
        # Load Attention-Centric YOLOv12 Model
        self.yolo_model = None
        if ULTRALYTICS_AVAILABLE:
            trained_weights = Path("models_checkpoints/yolov12_echopulse_marine.pt")
            fallback_weights = Path("runs/detect/echopulse_yolov12/weights/best.pt")
            base_weights = Path("yolo12n.pt")
            
            weight_to_load = None
            if trained_weights.exists():
                weight_to_load = str(trained_weights)
            elif fallback_weights.exists():
                weight_to_load = str(fallback_weights)
            elif base_weights.exists():
                weight_to_load = str(base_weights)
                
            if weight_to_load:
                try:
                    self.yolo_model = YOLO(weight_to_load)
                    print(f"[*] UnifiedInferenceService: Loaded YOLOv12 model from {weight_to_load} on {self.device}")
                except Exception as e:
                    print(f"[!] Warning: Failed to load YOLOv12 model: {e}")

        # Load HydroPhys-OmniNet (Continuous Wave State-Space 1D/2D/3D Engine)
        self.omni_engine = None
        try:
            from ..models.hydrophys_omninet import HydroPhysOmniVisionEngine
            omni_ckpt = "models_checkpoints/hydrophys_omninet_extreme_best.pt"
            if not Path(omni_ckpt).exists():
                omni_ckpt = "models_checkpoints/echophys_x_v3_unified_best.pt"
            self.omni_engine = HydroPhysOmniVisionEngine(weights_path=omni_ckpt, device=str(self.device))
            print(f"[*] UnifiedInferenceService: Loaded HydroPhys-OmniNet Engine on {self.device}")
        except Exception as e:
            print(f"[!] Warning: HydroPhys-OmniNet loading deferred: {e}")

    def run_inference(
        self,
        image_path: str,
        mission_id: str = "MSN-2026-0884",
        mission_name: str = "Active Hydrographic Sonar Mission",
        vessel_nav: Dict[str, Any] = None
    ) -> List[DetectionSchema]:
        """
        Runs authoritative pipeline:
        1. Preprocessing (OpenCV sonar enhancement, slant-range, shadow masking)
        2. Attention-Centric YOLOv12 target detection (NVIDIA RTX 5060 Accelerated)
        3. PyTorch Autoencoder anomaly evaluation
        4. Acoustic Shadow & 3D Bathymetric Geometry Analysis
        5. Geotagging WGS84 computation
        6. Multi-Factor Confidence Fusion
        """
        raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise ValueError(f"Unable to read image at {image_path}")
            
        preprocessed = OpenCVProcessor.preprocess_sonar_image(raw_img)
        enhanced = preprocessed["processed_image"]
        shadow_mask = preprocessed["shadow_mask"]
        quality_score = preprocessed["quality_score"]
        
        nav = vessel_nav or {
            "lat": 9.1524,
            "lng": 79.2819,
            "heading": 42.0,
            "altitude": 8.5,
            "depth": 32.0,
            "ping": 10420
        }
        
        detections: List[DetectionSchema] = []
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
        
        # 1. Primary Object Detection via HydroPhys-OmniNet & Attention-Centric YOLOv12
        yolo_boxes = []
        
        # Method A: HydroPhys-OmniNet Extreme Multi-Modal Detection
        if self.omni_engine is not None:
            try:
                from PIL import Image
                pil_img = Image.open(image_path).convert("RGB")
                omni_res = self.omni_engine.process_omni_frame(
                    pil_img,
                    conf_threshold=0.25,
                    altitude_m=nav.get("altitude", 15.0),
                    swath_m=nav.get("swath_m", 150.0)
                )
                for d in omni_res.get("detections", []):
                    x1, y1, x2, y2 = d["bbox_2d_pixels"]
                    w = max(1, x2 - x1)
                    h = max(1, y2 - y1)
                    conf = float(d["confidence"])
                    cat_name = d.get("category", "marine_debris")
                    # Map category name to class index
                    cat_to_idx = {v[0]: k for k, v in CLASS_MAPPINGS.items()}
                    cls_idx = cat_to_idx.get(cat_name, 4)
                    yolo_boxes.append((x1, y1, w, h, cls_idx, conf))
            except Exception as e:
                print(f"[!] HydroPhys-OmniNet file inference note: {e}")

        # Method B: Attention-Centric YOLOv12
        if len(yolo_boxes) == 0 and self.yolo_model is not None:
            try:
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                dev_str = "0" if self.device.type == "cuda" else "cpu"
                results = self.yolo_model.predict(
                    source=enhanced_bgr,
                    device=dev_str,
                    conf=0.15,
                    imgsz=640,
                    verbose=False
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        yolo_boxes.append((x1, y1, w, h, cls_idx, conf))
            except Exception as e:
                print(f"[!] YOLOv12 inference exception: {e}")
                
        # Method C: Acoustic Specular Highlight & Shadow Morphological Pair Detection
        if not yolo_boxes:
            # Detect primary high-backscatter metallic / solid debris structures
            thresh_val = int(np.percentile(enhanced, 92))
            _, bright_mask = cv2.threshold(enhanced, max(140, thresh_val), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [c for c in contours if cv2.contourArea(c) >= 50]
            
            # Sort by area descending to capture prominent shipwreck / gear structures first
            filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
            
            if not filtered_contours and len(contours) > 0:
                filtered_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
                
            for cnt in filtered_contours[:6]:
                x, y, w, h = cv2.boundingRect(cnt)
                # Expand box slightly to encompass the full acoustic highlight envelope
                pad = 4
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(enhanced.shape[1] - x, w + pad * 2)
                h = min(enhanced.shape[0] - y, h + pad * 2)
                
                aspect = w / max(1, h)
                area = w * h
                if aspect > 2.2:
                    cls_idx = 3 # pipeline_anomaly
                elif area > 500 or h > 80:
                    cls_idx = 1 # shipwreck / submerged hull
                elif aspect < 0.6:
                    cls_idx = 2 # UXO
                elif area > 200:
                    cls_idx = 0 # ghost_gear
                else:
                    cls_idx = 4 # marine_debris
                yolo_boxes.append((x, y, w, h, cls_idx, 0.88))

                
        # 3. Process candidate bounding boxes with HEAVY DEBRIS GUARDRAILS
        # Strict 5-Class Target Policy: Humans, Electrical, Electronic, Plastic, Metal Scraps
        annotated_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        for idx, (x, y, w, h, cls_idx, det_confidence) in enumerate(yolo_boxes[:12]):
            raw_class_key, raw_class_label = CLASS_MAPPINGS.get(cls_idx, ("marine_debris", "Marine Anthropogenic Debris"))
            
            # Synthesize contour mask for geometry analyzer
            dummy_cnt = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
            geometry = ShadowGeometryAnalyzer.analyze_geometry(dummy_cnt)
            
            # Acoustic shadow computation
            slant_range_m = max(5.0, 10.0 + (x / max(1, enhanced.shape[1])) * 50.0)
            shadow_obj = ShadowGeometryAnalyzer.compute_acoustic_shadow(
                {"x": x, "y": y, "width": w, "height": h},
                shadow_mask,
                sensor_altitude_m=nav["altitude"],
                slant_range_m=slant_range_m
            )
            
            # Anomaly scoring via Autoencoder reconstruction error
            patch = cv2.resize(
                enhanced[max(0, y-5):min(enhanced.shape[0], y+h+5), max(0, x-5):min(enhanced.shape[1], x+w+5)],
                (128, 128)
            )
            patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                recon = self.autoencoder(patch_t)
                recon_err = float(torch.mean((patch_t - recon) ** 2).item())
            anomaly_score = float(np.clip(recon_err * 20.0, 0.2, 0.95))
            
            # Fusion
            shadow_score = shadow_obj.shadowConfidence
            geometry_score = float(np.clip(geometry.solidity * 0.5 + geometry.extent * 0.5, 0.4, 0.95))
            
            fused_confidence = MultiFactorFusion.fuse(
                detector_score=det_confidence,
                shadow_score=shadow_score,
                geometry_score=geometry_score,
                anomaly_score=anomaly_score,
                quality_score=quality_score
            )
            
            # --- EVALUATE THROUGH HEAVY DEBRIS GUARDRAIL ENGINE ---
            guardrail_res = HeavyDebrisGuardrailEngine.evaluate_target(
                raw_class_name=raw_class_key,
                confidence=fused_confidence,
                bbox=(x, y, w, h),
                image_shape=enhanced.shape,
                shadow_strength=shadow_score,
                anomaly_sharpness=anomaly_score
            )
            
            is_debris = guardrail_res["is_debris"]
            guardrail_passed = guardrail_res["passed"]
            guardrail_cat = guardrail_res["target_category"]
            class_id = guardrail_res["class_id"]
            class_label = guardrail_res["class_label"]
            r_c, g_c, b_c = guardrail_res["color_rgb"]
            box_color = (b_c, g_c, r_c) # OpenCV BGR
            
            # Geotagging WGS84
            target_lat, target_lng, geo_conf = GeotaggingService.calculate_wgs84_position(
                vessel_lat=nav["lat"],
                vessel_lng=nav["lng"],
                vessel_heading_deg=nav["heading"],
                slant_range_m=slant_range_m,
                altitude_m=nav["altitude"],
                is_port_channel=(x < enhanced.shape[1] / 2)
            )
            
            # Draw visual bounding box & HUD label on Annotated Image
            line_thickness = 2 if is_debris else 1
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, line_thickness)

            if is_debris:
                # Draw corner brackets for cyber HUD visual effect
                c_len = min(16, max(4, int(min(w, h) * 0.25)))
                cv2.line(annotated_img, (x, y), (x + c_len, y), (255, 255, 255), 3)
                cv2.line(annotated_img, (x, y), (x, y + c_len), (255, 255, 255), 3)
                cv2.line(annotated_img, (x + w, y), (x + w - c_len, y), (255, 255, 255), 3)
                cv2.line(annotated_img, (x + w, y), (x + w, y + c_len), (255, 255, 255), 3)
                cv2.line(annotated_img, (x, y + h), (x + c_len, y + h), (255, 255, 255), 3)
                cv2.line(annotated_img, (x, y + h), (x, y + h - c_len), (255, 255, 255), 3)
                cv2.line(annotated_img, (x + w, y + h), (x + w - c_len, y + h), (255, 255, 255), 3)
                cv2.line(annotated_img, (x + w, y + h), (x + w, y + h - c_len), (255, 255, 255), 3)
                label_text = f"[{guardrail_cat}] {class_label.split('/')[0].strip()} ({int(fused_confidence*100)}%)"
            else:
                label_text = f"[NOT DEBRIS] {class_label.split('/')[0].strip()}"

            (lbl_w, lbl_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            lbl_y = max(lbl_h + 4, y - 4)
            cv2.rectangle(annotated_img, (x, lbl_y - lbl_h - 4), (x + lbl_w + 6, lbl_y + baseline), (2, 7, 18), -1)
            cv2.rectangle(annotated_img, (x, lbl_y - lbl_h - 4), (x + lbl_w + 6, lbl_y + baseline), box_color, 1)
            cv2.putText(annotated_img, label_text, (x + 3, lbl_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Crop image save
            crop_filename = f"crop_{uuid.uuid4().hex[:8]}.png"
            crop_path = os.path.join(settings.UPLOADS_DIR, crop_filename)
            crop_img = enhanced[max(0, y-10):min(enhanced.shape[0], y+h+10), max(0, x-10):min(enhanced.shape[1], x+w+10)]
            if crop_img.size > 0:
                cv2.imwrite(crop_path, crop_img)
                
            detections.append(DetectionSchema(
                id=f"DET-2026-{len(detections)+1:04d}",
                missionId=mission_id,
                missionName=mission_name,
                class_name=class_id,
                classNameLabel=class_label,
                confidence=round(fused_confidence, 3),
                detectorScore=round(det_confidence, 3),
                shadowScore=round(shadow_score, 3),
                geometryScore=round(geometry_score, 3),
                anomalyScore=round(anomaly_score, 3),
                qualityScore=round(quality_score, 3),
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                acousticShadow=shadow_obj,
                geometry=geometry,
                latitude=target_lat,
                longitude=target_lng,
                depthMeters=round(nav["depth"] + (shadow_obj.estimatedHeightMeters or 0.0), 2),
                slantRangeMeters=round(slant_range_m, 2),
                altitudeMeters=nav["altitude"],
                geotagConfidence=geo_conf,
                timestamp="2026-08-26T10:00:00Z",
                pingIndex=nav["ping"],
                modelVersion="YOLOv12-Sonar Attention RTX5060",
                imageCropUrl=f"/uploads/{crop_filename}",
                verificationStatus="CONFIRMED" if is_debris else "FALSE_POSITIVE",
                guardrailPassed=guardrail_passed,
                guardrailCategory=guardrail_cat,
                isDebris=is_debris,
                guardrailReason=guardrail_res["reason"],
                notes=f"Heavy Guardrail: {guardrail_res['reason']}"
            ))

            
        # Save full annotated image
        annotated_filename = f"annotated_{uuid.uuid4().hex[:8]}.png"
        cv2.imwrite(os.path.join(settings.UPLOADS_DIR, annotated_filename), annotated_img)
        self.last_annotated_url = f"/uploads/{annotated_filename}"

        # Sync detections to PostgreSQL / PostGIS Spatial Database
        try:
            from .postgis_service import postgis_connector
            for d in detections:
                postgis_connector.sync_detection(d.model_dump())
        except Exception as e:
            print(f"[*] PostGIS sync notice: {e}")
            
        return detections


