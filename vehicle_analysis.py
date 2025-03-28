import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import time
import os
from sort import Sort

class VehicleAnalyzer:
    def __init__(self):
        """Initialize Vehicle Analyzer"""
        try:
            # Initialize YOLO with your custom model
            self.model = YOLO(r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\best(3).pt")
            self.conf_threshold = 0.5
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")
            
            # Initialize tracker
            self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
            
            # Initialize vehicle counters with correct classes
            self.unique_vehicles = {}  # {track_id: vehicle_class}
            self.vehicle_counts = {
                'biker': set(),
                'bus': set(),
                'motobike': set(),
                'pedestrian': set(),
                'sedan': set(),
                'taxi': set(),
                'truck': set()
            }
            
            # Class mapping for your custom model
            self.class_mapping = {
                0: 'biker',
                1: 'bus',
                2: 'motobike',
                3: 'pedestrian',
                4: 'sedan',
                5: 'taxi',
                6: 'truck'
            }
            
            # Results dictionary
            self.results = {
                'total_counts': {k: 0 for k in self.vehicle_counts.keys()},
                'processing_time': 0,
                'frames_processed': 0
            }
            
        except Exception as e:
            raise Exception(f"Initialization error: {str(e)}")

    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get unique color for each vehicle class"""
        colors = {
            'biker': (255, 0, 0),     # Blue
            'bus': (0, 255, 0),       # Green
            'motobike': (0, 0, 255),  # Red
            'pedestrian': (255, 255, 0), # Cyan
            'sedan': (255, 0, 255),   # Magenta
            'taxi': (0, 255, 255),    # Yellow
            'truck': (128, 128, 128)  # Gray
        }
        return colors.get(class_name, (200, 200, 200))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame"""
        try:
            # Run YOLO detection
            results = self.model(frame)[0]
            processed_frame = frame.copy()
            
            # Prepare detections for tracker
            detections = []
            classes = []
            
            # Process YOLO detections
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                
                if cls in self.class_mapping and conf >= self.conf_threshold:
                    detections.append([x1, y1, x2, y2, conf])
                    classes.append(cls)
            
            # Update tracker
            if len(detections) > 0:
                tracked_objects = self.tracker.update(np.array(detections))
            else:
                tracked_objects = np.empty((0, 5))
            
            # Process tracked objects
            frame_counts = {class_name: 0 for class_name in self.vehicle_counts.keys()}
            
            for i, track in enumerate(tracked_objects):
                if i >= len(classes):
                    continue
                    
                track_id = int(track[4])
                bbox = track[:4]
                cls = classes[i]
                vehicle_class = self.class_mapping[cls]
                
                # Update unique vehicle counts
                if track_id not in self.unique_vehicles:
                    self.unique_vehicles[track_id] = vehicle_class
                    self.vehicle_counts[vehicle_class].add(track_id)
                
                # Draw bounding box
                color = self.get_color_for_class(vehicle_class)
                cv2.rectangle(
                    processed_frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2
                )
                
                # Add label
                label = f"{vehicle_class} ID:{track_id}"
                cv2.putText(
                    processed_frame,
                    label,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                
                frame_counts[vehicle_class] += 1
            
            # Add count overlay
            processed_frame = self.add_count_overlay(processed_frame)
            return processed_frame, frame_counts
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame, {class_name: 0 for class_name in self.vehicle_counts.keys()}

    def add_count_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add count overlay to frame"""
        try:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            y_offset = 30
            cv2.putText(frame, "Unique Vehicle Counts:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for vehicle_class, ids in self.vehicle_counts.items():
                y_offset += 20
                text = f"{vehicle_class}: {len(ids)}"
                cv2.putText(frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           self.get_color_for_class(vehicle_class), 2)
            
            return frame
        except Exception as e:
            print(f"Overlay error: {str(e)}")
            return frame

    def process_video(self, video_path: str):
        """Process video and save results"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        output_path = f"CarComp_{Path(video_path).name}"
        start_time = time.time()
        
        try:
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
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, _ = self.process_frame(frame)
                out.write(processed_frame)
                
                cv2.imshow('Vehicle Analysis', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
        finally:
            # Save results
            self.results['processing_time'] = time.time() - start_time
            self.results['frames_processed'] = frame_count
            
            # Cleanup
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            self.print_statistics()
            return self.results

    def print_statistics(self):
        """Print final statistics"""
        total_vehicles = sum(len(ids) for ids in self.vehicle_counts.values())
        
        print("\nVehicle Detection Statistics")
        print("==========================")
        print(f"Total Unique Vehicles: {total_vehicles}")
        print(f"Processing Time: {self.results['processing_time']:.2f} seconds")
        print(f"Frames Processed: {self.results['frames_processed']}")
        print("\nVehicle Class Distribution:")
        
        for vehicle_class, ids in self.vehicle_counts.items():
            count = len(ids)
            percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
            print(f"{vehicle_class}: {count} ({percentage:.1f}%)")

def main():
    try:
        # Initialize analyzer
        analyzer = VehicleAnalyzer()
        
        # Get video path
        video_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\abc.mp4"
        
        # Process video
        results = analyzer.process_video(video_path)
        print("\nProcessing completed successfully!")
        return results
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()