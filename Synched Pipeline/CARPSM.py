import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import os
import csv
from datetime import datetime
from tracker import Sort

class CarPSMDetector:
    def __init__(self, model_path, csv_filename):
        """Initialize CarPSMDetector with model path and CSV file path"""
        self.csv_filename = csv_filename
        print(f"Initializing CarPSMDetector with CSV file: {csv_filename}")
        
        # Initialize storage for car detections and lanes
        self.car_detections = {}  # track_id -> (vehicle_type, lane_num, timestamp)
        self.lanes = []
        self.target_y = None
        self.pipeline = None  # Will store reference to pipeline
        
        # Setup CSV file
        self.setup_csv_file()
        
        print("CarPSMDetector initialization complete.")

    def setup_csv_file(self):
        """Setup CSV file for logging vehicle detections"""
        os.makedirs('logs', exist_ok=True)
        headers = ['Vehicle_ID', 'Vehicle_Type', 'Lane_Number', 'Detection_Time']
        
        try:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            print(f"Error setting up CSV file: {str(e)}")
            raise

    def set_pipeline(self, pipeline):
        """Set reference to the main pipeline"""
        self.pipeline = pipeline
        print("Pipeline reference set in CARPSM detector")

    def select_target_y(self, frame):
        """Allow user to select target y-coordinate"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.target_y = y
                frame_copy = frame.copy()
                cv2.line(frame_copy, (0, y), (frame.shape[1], y), (0, 0, 255), 2)
                cv2.imshow('Select Target Y-Coordinate', frame_copy)
                cv2.waitKey(1000)
                cv2.destroyWindow('Select Target Y-Coordinate')

        cv2.namedWindow('Select Target Y-Coordinate')
        cv2.setMouseCallback('Select Target Y-Coordinate', mouse_callback)
        
        print("\nClick to select target y-coordinate for vehicle detection")
        cv2.imshow('Select Target Y-Coordinate', frame)
        while self.target_y is None:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    def get_lane_number(self, point):
        """Determine which lane a point is in"""
        for i, lane_points in enumerate(self.lanes):
            polygon = np.array(lane_points)
            if self.point_in_polygon(point, polygon):
                return i + 1
        return None

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            if ((polygon[i][1] > y) != (polygon[j][1] > y) and
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                 (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
                inside = not inside
            j = i
            
        return inside

    def get_vehicle_type(self, track_id, bbox):
        """Determine vehicle type based on class or size"""
        if self.pipeline and hasattr(self.pipeline, 'vehicle_classes') and track_id in self.pipeline.vehicle_classes:
            cls = self.pipeline.vehicle_classes[track_id]
            if cls == 0:
                return "Biker"
            elif cls == 2:
                return "Motorbike"
            elif cls == 4:
                return "Car"
            elif cls == 5:
                return "Taxi"
        
        # Fallback to size-based classification
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        if area < 5000:  # Small vehicles
            return "Motorbike"  # Changed from Biker to be consistent
        elif aspect_ratio > 1.5:  # Wide vehicles
            return "Car"
        return "Unknown"

    def process_frame(self, frame, tracks, current_time):
        """Process a frame and return annotated frame"""
        # Create copy of frame for drawing
        car_frame = frame.copy()
        
        # Draw lane polygons with transparency
        overlay = car_frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Different color for each lane
        
        # Draw lanes
        for i, lane_points in enumerate(self.lanes):
            try:
                polygon = np.array([lane_points], dtype=np.int32)
                cv2.fillPoly(overlay, polygon, colors[i])
                # Add lane number
                center = np.mean(lane_points, axis=0).astype(int)
                cv2.putText(car_frame, f"Lane {i+1}", tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error drawing lane {i+1}: {str(e)}")
        
        # Blend the overlay with original frame
        alpha = 0.3
        car_frame = cv2.addWeighted(overlay, alpha, car_frame, 1 - alpha, 0)
        
        # Draw target y-coordinate line with dashed effect
        if self.target_y is not None:
            for x in range(0, car_frame.shape[1], 20):
                cv2.line(car_frame, (x, self.target_y), (x + 10, self.target_y),
                        (0, 0, 255), 2)
        
        # Process tracks
        for track in tracks:
            try:
                track_id = int(track[4])
                bbox = track[:4]
                center_x = int((float(bbox[0]) + float(bbox[2])) / 2)
                center_y = int((float(bbox[1]) + float(bbox[3])) / 2)
                
                # Check if vehicle crosses target y-coordinate
                if (self.target_y is not None and 
                    track_id not in self.car_detections and 
                    abs(center_y - self.target_y) < 5):  # 5-pixel threshold
                    
                    lane_num = self.get_lane_number((center_x, center_y))
                    if lane_num is not None:
                        vehicle_type = self.get_vehicle_type(track_id, bbox)
                        self.car_detections[track_id] = (vehicle_type, lane_num, current_time)
                        print(f"{vehicle_type} {track_id} detected in lane {lane_num} at {current_time}")
                        # Write to CSV
                        self.update_csv(track_id, vehicle_type, lane_num, current_time)
                
                # Draw vehicle visualization
                if track_id in self.car_detections:
                    vehicle_type, lane_num, timestamp = self.car_detections[track_id]
                    color = (0, 255, 0) if vehicle_type == "Car" else (0, 255, 255) if vehicle_type == "Biker" else (255, 0, 0)
                else:
                    color = (0, 255, 0)  # Default color
                    vehicle_type = self.get_vehicle_type(track_id, bbox)  # Get type even if not crossed line
                    lane_num = None
                    timestamp = None
                
                cv2.rectangle(car_frame, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), color, 2)
                
                # Draw ID and detection info
                text_y = int(bbox[1]) - 10
                text = f"ID: {track_id} ({vehicle_type})"
                cv2.putText(car_frame, text, (int(bbox[0]), text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if lane_num is not None:
                    text_y -= 20
                    cv2.putText(car_frame, f"Lane: {lane_num}", (int(bbox[0]), text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    text_y -= 20
                    cv2.putText(car_frame, f"Time: {timestamp}", (int(bbox[0]), text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw center point
                cv2.circle(car_frame, (center_x, center_y), 3, (255, 0, 0), -1)
                
            except Exception as e:
                print(f"Error processing track: {str(e)}")
                continue
        
        return car_frame

    def update_csv(self, track_id, vehicle_type, lane_num, timestamp):
        """Update CSV with vehicle detection"""
        if self.csv_filename:
            try:
                with open(self.csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([track_id, vehicle_type, lane_num, timestamp])
            except Exception as e:
                print(f"Error writing to vehicle CSV: {str(e)}") 