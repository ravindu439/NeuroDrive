
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class VehicleDetector:
    
    def __init__(self, model_path, confidence_threshold=0.25):

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def detect(self, image_path):

        # Run inference
        results = self.model(image_path, conf=self.confidence_threshold)[0]
        
        # Parse results
        detections = {
            'boxes': [],
            'labels': [],
            'confidences': [],
            'class_counts': {}
        }
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = results.names[cls]
                
                detections['boxes'].append([int(x1), int(y1), int(x2), int(y2)])
                detections['labels'].append(label)
                detections['confidences'].append(conf)
                
                # Update class counts
                if label in detections['class_counts']:
                    detections['class_counts'][label] += 1
                else:
                    detections['class_counts'][label] = 1
        
        return detections
    
    def annotate_image(self, image_path, detections, output_path):

        # Load image
        image = cv2.imread(str(image_path))
        
        # Define colors for all 8 classes in the dataset
        # Colors in BGR format (OpenCV uses BGR)
        colors = {
            'bicycle': (0, 165, 255),      # Orange
            'bus': (255, 0, 255),          # Magenta
            'car': (0, 255, 0),            # Green
            'motorcycle': (255, 191, 0),   # Deep Sky Blue
            'three_wheeler': (0, 255, 255),# Yellow
            'tractor': (42, 42, 165),      # Brown
            'truck': (255, 0, 0),          # Blue
            'van': (255, 255, 0),          # Cyan
        }
        default_color = (128, 128, 128)  # Gray for any unknown class
        
        # Draw detections
        for box, label, conf in zip(detections['boxes'], detections['labels'], detections['confidences']):
            x1, y1, x2, y2 = box
            
            # Get color for this class
            color = colors.get(label.lower(), default_color)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            text = f"{label}: {conf:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Save annotated image
        cv2.imwrite(str(output_path), image)
        return str(output_path)
    
    def process_single_image(self, image_path, output_dir):

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run detection
        detections = self.detect(image_path)
        
        # Generate output filename
        filename = Path(image_path).name
        output_path = Path(output_dir) / f"annotated_{filename}"
        
        # Annotate image
        annotated_path = self.annotate_image(image_path, detections, output_path)
        
        return annotated_path, detections
    
    def process_folder(self, folder_path, output_dir, organize_by_class=True):

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Get all image files
        image_files = [
            f for f in Path(folder_path).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        results = []
        
        for image_path in image_files:
            # Run detection
            detections = self.detect(str(image_path))
            
            # Determine dominant class
            dominant_class = 'unknown'
            if detections['class_counts']:
                # Get class with highest count
                dominant_class = max(
                    detections['class_counts'].items(),
                    key=lambda x: x[1]
                )[0]
            
            # Determine output path
            if organize_by_class:
                class_dir = Path(output_dir) / dominant_class
                os.makedirs(class_dir, exist_ok=True)
                output_path = class_dir / f"annotated_{image_path.name}"
            else:
                output_path = Path(output_dir) / f"annotated_{image_path.name}"
            
            # Annotate image
            annotated_path = self.annotate_image(str(image_path), detections, output_path)
            
            # Store results
            results.append({
                'image_name': image_path.name,
                'annotated_path': annotated_path,
                'detections': detections,
                'dominant_class': dominant_class,
                'total_vehicles': len(detections['labels']),
                'avg_confidence': np.mean(detections['confidences']) if detections['confidences'] else 0
            })
        
        return results
