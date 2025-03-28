import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
from typing import Dict, List, Tuple

class VehicleAnalyzer:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize Vehicle Analyzer
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\best(3).pt")
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize vehicle counters
        self.vehicle_counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0
        }
        
        # Class mapping (adjust based on your model's classes)
        self.class_mapping = {
            2: 'car',
            7: 'truck',
            5: 'bus',
            3: 'motorcycle',
            1: 'bicycle'
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame and return annotated frame with detection counts"""
        # Run YOLO detection
        results = self.model(frame)[0]
        
        # Process detections
        processed_frame = frame.copy()
        frame_counts = {class_name: 0 for class_name in self.vehicle_counts.keys()}
        
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            
            if cls in self.class_mapping and conf >= self.conf_threshold:
                vehicle_class = self.class_mapping[cls]
                
                # Update counts
                self.vehicle_counts[vehicle_class] += 1
                frame_counts[vehicle_class] += 1
                
                # Draw bounding box
                color = self.get_color_for_class(vehicle_class)
                cv2.rectangle(
                    processed_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )
                
                # Add label
                label = f"{vehicle_class}: {conf:.2f}"
                cv2.putText(
                    processed_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        # Add count overlay
        processed_frame = self.add_count_overlay(processed_frame, frame_counts)
        return processed_frame, frame_counts
    
    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Return unique color for each vehicle class"""
        colors = {
            'car': (0, 255, 0),      # Green
            'truck': (255, 0, 0),    # Blue
            'bus': (0, 0, 255),      # Red
            'motorcycle': (255, 255, 0), # Cyan
            'bicycle': (255, 0, 255)  # Magenta
        }
        return colors.get(class_name, (128, 128, 128))
    
    def add_count_overlay(self, frame: np.ndarray, counts: Dict) -> np.ndarray:
        """Add count information overlay to frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Add text
        y_offset = 30
        cv2.putText(frame, "Vehicle Counts:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for vehicle_class, count in counts.items():
            y_offset += 20
            total_count = self.vehicle_counts[vehicle_class]
            text = f"{vehicle_class}: {count} (Total: {total_count})"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.get_color_for_class(vehicle_class), 2)
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = 'output.mp4'):
        """Process video file and save results"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        print("Processing video... Press 'q' to stop")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, _ = self.process_frame(frame)
                
                # Write frame
                out.write(processed_frame)
                
                # Display live processing
                cv2.imshow('Vehicle Analysis', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
            # Print final statistics
            self.print_statistics()
            
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
    
    def print_statistics(self):
        """Print final vehicle detection statistics"""
        total_vehicles = sum(self.vehicle_counts.values())
        
        print("\nVehicle Detection Statistics")
        print("==========================")
        print(f"Total Vehicles Detected: {total_vehicles}")
        
        for vehicle_class, count in self.vehicle_counts.items():
            percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
            print(f"{vehicle_class}: {count} ({percentage:.1f}%)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Vehicle Detection and Analysis')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    
    # Initialize and run analyzer
    analyzer = VehicleAnalyzer(args.model, args.conf)
    analyzer.process_video(args.video, args.output)

if __name__ == "__main__":
    main() 