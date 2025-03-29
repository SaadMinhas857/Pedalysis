from typing import Tuple, Dict
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import os
import time
import sys
from tracker import Sort  # Using the provided tracker instead of SORT

class VehicleAnalyzer:
    def __init__(self):
        """Initialize Vehicle Analyzer"""
        try:
            # Load YOLO model
            model_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\best(3).pt"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            self.model = YOLO(model_path)
            
            # Initialize tracker
            self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
            
            # Store vehicle classes
            self.vehicle_classes = {}
            
            # Initialize counters
            self.vehicle_counts = {
                'biker': set(),
                'bus': set(),
                'motobike': set(),
                'pedestrian': set(),
                'sedan': set(),
                'taxi': set(),
                'truck': set()
            }
            
            # Class mapping
            self.class_mapping = {
                0: 'biker',
                1: 'bus',
                2: 'motobike',
                3: 'pedestrian',
                4: 'sedan',
                5: 'taxi',
                6: 'truck'
            }
            
            print("Initialization complete. Ready to process video.")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def get_color_for_class(self, class_name):
        colors = {
            'biker': (255, 0, 0),
            'bus': (0, 255, 0),
            'motobike': (0, 0, 255),
            'pedestrian': (255, 255, 0),
            'sedan': (255, 0, 255),
            'taxi': (0, 255, 255),
            'truck': (128, 128, 128)
        }
        return colors.get(class_name, (200, 200, 200))

    def process_video(self, video_path):
        """Process video file"""
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        try:
            print(f"Starting video processing: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create output directory if it doesn't exist
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output video writer
            output_path = os.path.join(output_dir, f"CarComp_{os.path.basename(video_path)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            print("Processing frames...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(frame)[0]
                
                # Process detections
                detections = []
                classes = []
                
                # Get all detections without confidence filtering
                for det in results.boxes.data.tolist():
                    x1, y1, x2, y2, _, cls = det  # Ignoring confidence value
                    cls = int(cls)
                    detections.append([x1, y1, x2, y2, 1.0])  # Adding dummy confidence of 1.0
                    classes.append(cls)
                
                # Convert detections to numpy array
                if len(detections) > 0:
                    detections = np.array(detections)
                    # Update tracker
                    tracked_objects = self.tracker.update(detections)
                else:
                    tracked_objects = np.empty((0, 5))
                
                # Process tracked objects
                for track in tracked_objects:
                    bbox = track[:4]
                    track_id = int(track[4])
                    
                    # Get the class for this detection
                    if track_id not in self.vehicle_classes and len(classes) > 0:
                        # Assign the class for new tracks
                        det_idx = np.argmin(np.sum(np.abs(detections[:, :4] - bbox), axis=1))
                        vehicle_class = self.class_mapping[classes[det_idx]]
                        self.vehicle_classes[track_id] = vehicle_class
                        self.vehicle_counts[vehicle_class].add(track_id)
                    
                    if track_id in self.vehicle_classes:
                        vehicle_class = self.vehicle_classes[track_id]
                        
                        # Draw bounding box
                        color = self.get_color_for_class(vehicle_class)
                        cv2.rectangle(
                            frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color,
                            2
                        )
                        
                        # Add label
                        label = f"{vehicle_class} ID:{track_id}"
                        cv2.putText(
                            frame,
                            label,
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
                
                # Add count overlay
                self.add_count_overlay(frame)
                
                # Write frame
                out.write(frame)
                
                # Display frame
                cv2.imshow('Vehicle Analysis', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            raise
            
        finally:
            # Cleanup
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            if frame_count > 0:
                processing_time = time.time() - start_time
                print("\nProcessing Complete!")
                print(f"Total frames processed: {frame_count}")
                print(f"Processing time: {processing_time:.2f} seconds")
                print("\nVehicle Counts:")
                for vehicle_class, ids in self.vehicle_counts.items():
                    print(f"{vehicle_class}: {len(ids)}")

    def add_count_overlay(self, frame):
        y_offset = 30
        cv2.putText(frame, "Vehicle Counts:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for vehicle_class, ids in self.vehicle_counts.items():
            y_offset += 20
            text = f"{vehicle_class}: {len(ids)}"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.get_color_for_class(vehicle_class), 2)

def main():
    try:
        # Create analyzer
        analyzer = VehicleAnalyzer()
        
        # Process video
        video_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\abc.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            print("Please check the video path and try again.")
            return
        
        analyzer.process_video(video_path)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 