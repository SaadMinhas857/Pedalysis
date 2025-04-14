import cv2
import numpy as np
import time
import pandas as pd
import csv
from datetime import datetime
from collections import defaultdict, deque
import supervision as sv
from ultralytics import YOLO

class UltralyticsProcessor:
    def __init__(self, csv_filename: str, model_path: str = "best(3).pt", conf_threshold: float = 0.5):
        """
        Initialize the Ultralytics Processor
        
        Args:
            csv_filename (str): Path to CSV file for logging results
            model_path (str): Path to the YOLOv8 model weights
            conf_threshold (float): Confidence threshold for detections
        """
        try:
            # Store CSV filename
            self.csv_filename = csv_filename
            print(f"Initializing UltralyticsProcessor with CSV file: {csv_filename}")
            
            # Initialize YOLO model
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            print(f"Model loaded from: {model_path}")
            
            # Initialize tracking variables
            self.object_tracking = {}  # {id: {'positions': [], 'times': [], 'current_speed': None}}
            self.object_speeds = {}    # {id: speed}
            
            # Initialize CSV data
            self.csv_data = {'id': [], 'class': [], 'speed': [], 'timestamp': []}
            
            # Setup CSV file
            self.setup_csv_file()
            
            # Initialize ByteTrack
            self.byte_track = sv.ByteTrack(
                frame_rate=30,  # Default FPS, will be updated when video is loaded
                track_activation_threshold=conf_threshold
            )
            
            # Initialize annotators
            self.box_annotator = None
            self.label_annotator = None
            self.trace_annotator = None
            
            # Initialize coordinates history
            self.coordinates = defaultdict(lambda: deque(maxlen=60))  # 2 seconds history at 30fps
            
            print("UltralyticsProcessor initialization complete.")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
    
    def setup_csv_file(self):
        """Setup CSV file with headers"""
        try:
            # Create CSV file with headers
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id', 'class', 'speed', 'timestamp'])
            print(f"CSV file created: {self.csv_filename}")
        except Exception as e:
            print(f"Error creating CSV file: {str(e)}")
    
    def select_reference_points(self, frame):
        """Allow user to select reference points for speed calculation"""
        points = []
        frame_copy = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(frame_copy, points[-2], points[-1], (0, 255, 0), 2)
                cv2.imshow("Select Reference Points", frame_copy)
        
        cv2.namedWindow("Select Reference Points", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Reference Points", mouse_callback)
        
        print("\nInstructions:")
        print("1. Click to select 2 points to define the reference line for speed calculation")
        print("2. Press 'q' when done or 'r' to reset")
        
        while True:
            cv2.imshow("Select Reference Points", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(points) == 2:
                break
            elif key == ord('r'):
                points = []
                frame_copy = frame.copy()
                cv2.imshow("Select Reference Points", frame_copy)
        
        cv2.destroyAllWindows()
        
        if len(points) == 2:
            self.reference_line = points
            print(f"Reference line selected: {self.reference_line}")
            return True
        else:
            print("Not enough points selected. Using default reference line.")
            # Set default coordinates
            self.reference_line = [(100, 300), (700, 300)]
            return False
    
    def calculate_speed(self, elapsed_time, distance):
        """Calculate speed in km/h from elapsed time and distance"""
        if elapsed_time > 0:
            speed_ms = distance / elapsed_time
            return speed_ms * 3.6  # Convert to km/h
        return 0
    
    def update_csv_data(self, object_id, class_id, speed):
        """Update CSV data for an object"""
        if object_id not in self.csv_data['id']:
            self.csv_data['id'].append(object_id)
            self.csv_data['class'].append(class_id)
            self.csv_data['speed'].append(speed)
            self.csv_data['timestamp'].append(datetime.now())
            print(f"Added/Updated object {object_id} to CSV with speed: {speed} km/h")
        else:
            # Update existing entry
            idx = self.csv_data['id'].index(object_id)
            self.csv_data['speed'][idx] = speed
            self.csv_data['timestamp'][idx] = datetime.now()
    
    def has_crossed_line(self, prev_pos, curr_pos):
        """Check if an object has crossed the reference line"""
        if not hasattr(self, 'reference_line') or len(self.reference_line) != 2:
            return False
            
        # Get line endpoints
        p1, p2 = self.reference_line
        
        # Check if the line segment intersects with the reference line
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        # Check if the line segments intersect
        return ccw(prev_pos, p1, p2) != ccw(curr_pos, p1, p2) and \
               ccw(prev_pos, curr_pos, p1) != ccw(prev_pos, curr_pos, p2)

    def process_frame(self, frame, vehicle_tracks, visible_ids):
        """Process a frame with tracked objects"""
        processed_frame = frame.copy()
        
        # Draw reference line if it exists
        if hasattr(self, 'reference_line'):
            p1, p2 = self.reference_line
            cv2.line(processed_frame, p1, p2, (0, 255, 255), 2)
            cv2.putText(processed_frame, "Speed Measurement Line", 
                       (p1[0], p1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            # Draw additional reference lines
            if hasattr(self, 'reference_lines'):
                colors = [(0, 255, 0), (255, 128, 0), (255, 0, 0)]  # Green, Orange, Red
                labels = ["15m Line", "10m Line", "5m Line"]
                for i, (line, color, label) in enumerate(zip(self.reference_lines, colors, labels)):
                    p1, p2 = line
                    cv2.line(processed_frame, p1, p2, color, 2)
                    cv2.putText(processed_frame, label, 
                               (p1[0], p1[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Process each track
        for track in vehicle_tracks:
            track_id = int(track[4])
            
            # Skip if not in visible IDs
            if track_id not in visible_ids:
                continue
            
            # Get track position
            x1, y1, x2, y2 = map(int, track[:4])
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Draw bounding box and ID
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_frame, f"ID: {track_id}", 
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Get speeds from vehicle_speeds dictionary
            if hasattr(self, 'vehicle_speeds') and track_id in self.vehicle_speeds:
                speeds = self.vehicle_speeds[track_id]
                
                # Calculate average speed if we have all measurements
                if len(speeds) >= 4:  # We have speeds for all lines
                    avg_speed = sum(speeds.values()) / len(speeds)
                    
                    # Store speeds in database
                    speed_data = {
                        1: speeds.get('main', 0),  # Reference line speed
                        2: speeds.get('5m', 0),    # 5m line speed
                        3: speeds.get('10m', 0),   # 10m line speed
                        4: speeds.get('15m', 0),   # 15m line speed
                        5: avg_speed               # Average speed
                    }
                    
                    # Get vehicle class
                    if hasattr(self.pipeline, 'vehicle_classes') and track_id in self.pipeline.vehicle_classes:
                        class_id = self.pipeline.vehicle_classes[track_id]
                    else:
                        class_id = None
                    
                    # Insert data into database
                    self.pipeline.insert_track_data(track_id, 'vehicle', speed_data, class_id)
                    
                    # Draw speeds on frame
                    y_offset = y1 - 30
                    for line_key, speed in speeds.items():
                        cv2.putText(processed_frame, f"{line_key}: {speed:.1f} km/h",
                                  (x1, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset -= 20
                    
                    # Draw average speed
                    cv2.putText(processed_frame, f"Avg: {avg_speed:.1f} km/h",
                              (x1, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return processed_frame
    
    def save_results_to_csv(self):
        """Save results to CSV file"""
        try:
            # Save the speeds to CSV
            if self.csv_data['id']:
                df = pd.DataFrame(self.csv_data)
                df.to_csv(self.csv_filename, index=False)
                print(f"\nObject speed results saved to: {self.csv_filename}")
            else:
                print("No object speed data to save in CSV.")
                
        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")
    
    def print_statistics(self):
        """Print final statistics"""
        total_objects = len(self.csv_data['id'])
        
        print("\nUltralytics Processor Statistics")
        print("================================")
        print(f"Total Unique Objects: {total_objects}")
        
        if total_objects > 0:
            print("\nSpeed Statistics:")
            speeds = [s for s in self.csv_data['speed'] if s is not None]
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
                print(f"Average Speed: {avg_speed:.1f} km/h")
                print(f"Maximum Speed: {max_speed:.1f} km/h")
            
            # Count objects by class
            class_counts = {}
            for class_id in self.csv_data['class']:
                if class_id is not None:
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    else:
                        class_counts[class_id] = 1
            
            print("\nObject Counts by Class:")
            for class_id, count in class_counts.items():
                print(f"Class {class_id}: {count} objects") 