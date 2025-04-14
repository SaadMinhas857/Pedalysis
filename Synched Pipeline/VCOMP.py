import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time
import os
import csv
from datetime import datetime

class VehicleAnalyzer:
    def __init__(self, csv_filename: str):
        """Initialize Vehicle Analyzer"""
        try:
            # Store CSV filename
            self.csv_filename = csv_filename
            print(f"Initializing VehicleAnalyzer with CSV file: {csv_filename}")
            
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
            
            # Initialize time-based counting
            self.minute_counts = []  # List to store per-minute counts
            self.current_minute = 0
            self.current_minute_counts = {
                'biker': set(),
                'bus': set(),
                'motobike': set(),
                'pedestrian': set(),
                'sedan': set(),
                'taxi': set(),
                'truck': set()
            }
            self.last_time = 0
            
            # Region selection
            self.count_region = None  # Will store the points of the counting region
            
            # Class mapping for your custom model
            self.class_mapping = {
                0: 'biker',     # Biker
                1: 'bus',       # Bus
                2: 'motobike',  # Motobike
                3: 'pedestrian', # Pedestrian
                4: 'sedan',     # Sedan
                5: 'taxi',      # Taxi
                6: 'truck'      # Truck
            }
            
            # Setup CSV file
            self.setup_csv_file()
            
            print("VehicleAnalyzer initialization complete.")
            
        except Exception as e:
            raise Exception(f"Initialization error: {str(e)}")

    def setup_csv_file(self):
        """Setup CSV file with headers"""
        os.makedirs('logs', exist_ok=True)
        headers = ['Time Period', 'Vehicle Class', 'Count', 'Percentage']
        
        try:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            print(f"Error setting up CSV file: {str(e)}")
            raise

    def select_count_region(self, frame):
        """Allow user to select region for counting vehicles"""
        points = []
        frame_copy = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(frame_copy, points[-2], points[-1], (0, 255, 0), 2)
                if len(points) == 4:
                    cv2.line(frame_copy, points[-1], points[0], (0, 255, 0), 2)
                cv2.imshow("Select Count Region", frame_copy)
        
        cv2.namedWindow("Select Count Region", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Count Region", mouse_callback)
        
        print("\nSelect 4 points to define the vehicle counting region")
        print("Press 'q' when done or 'r' to reset")
        
        while True:
            cv2.imshow("Select Count Region", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(points) == 4:
                break
            elif key == ord('r'):
                points = []
                frame_copy = frame.copy()
                cv2.imshow("Select Count Region", frame_copy)
        
        cv2.destroyAllWindows()
        self.count_region = np.array(points)
        return points

    def is_point_in_region(self, point):
        """Check if a point is inside the counting region"""
        if self.count_region is None:
            return True  # If no region defined, count everywhere
        return cv2.pointPolygonTest(self.count_region, point, False) >= 0

    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get unique color for each vehicle class"""
        colors = {
            'biker': (255, 0, 0),      # Blue
            'bus': (0, 255, 0),        # Green
            'motobike': (0, 0, 255),   # Red
            'pedestrian': (255, 255, 0), # Cyan
            'sedan': (255, 0, 255),    # Magenta
            'taxi': (0, 255, 255),     # Yellow
            'truck': (128, 128, 128)   # Gray
        }
        return colors.get(class_name, (200, 200, 200))

    def process_frame(self, frame: np.ndarray, tracks: np.ndarray, vehicle_classes: Dict[int, int], frame_time: float) -> np.ndarray:
        """Process a frame with tracked objects and their classes"""
        try:
            processed_frame = frame.copy()
            frame_counts = {class_name: 0 for class_name in self.vehicle_counts.keys()}
            
            # Check if a minute has passed
            current_minute = int(frame_time // 60)
            if current_minute > self.current_minute:
                # Save the previous minute's counts
                self.minute_counts.append(self.current_minute_counts)
                # Reset for new minute
                self.current_minute_counts = {
                    'biker': set(),
                    'bus': set(),
                    'motobike': set(),
                    'pedestrian': set(),
                    'sedan': set(),
                    'taxi': set(),
                    'truck': set()
                }
                self.current_minute = current_minute
            
            # Draw counting region if defined
            if self.count_region is not None:
                cv2.polylines(processed_frame, [self.count_region], True, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Counting Region", 
                           tuple(self.count_region[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process tracked objects
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                center_point = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                
                # Only process if in counting region
                if not self.is_point_in_region(center_point):
                    continue
                
                # Get vehicle class from pipeline's vehicle_classes
                if track_id in vehicle_classes:
                    cls = vehicle_classes[track_id]
                    vehicle_class = self.class_mapping[cls]
                    
                    # Update unique vehicles count
                    if track_id not in self.unique_vehicles:
                        self.unique_vehicles[track_id] = vehicle_class
                        self.vehicle_counts[vehicle_class].add(track_id)
                        # Add to current minute counts
                        self.current_minute_counts[vehicle_class].add(track_id)
                    
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
            
            # Add count overlay with time period
            processed_frame = self.add_count_overlay(processed_frame, frame_time)
            return processed_frame
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame

    def add_count_overlay(self, frame: np.ndarray, frame_time: float) -> np.ndarray:
        """Add count overlay to frame"""
        try:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 250), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            y_offset = 30
            minutes = int(frame_time // 60)
            seconds = int(frame_time % 60)
            cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 20
            cv2.putText(frame, f"Minute {self.current_minute + 1} Counts:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for vehicle_class, ids in self.current_minute_counts.items():
                y_offset += 20
                text = f"{vehicle_class}: {len(ids)}"
                cv2.putText(frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           self.get_color_for_class(vehicle_class), 2)
            
            return frame
        except Exception as e:
            print(f"Overlay error: {str(e)}")
            return frame

    def save_results_to_csv(self):
        """Save results to CSV file"""
        try:
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Time Period', 'Vehicle Class', 'Count', 'Percentage'])
                
                # Write per-minute statistics
                for minute, counts in enumerate(self.minute_counts):
                    total = sum(len(ids) for ids in counts.values())
                    for vehicle_class, ids in counts.items():
                        count = len(ids)
                        percentage = (count / total * 100) if total > 0 else 0
                        writer.writerow([f"Minute {minute + 1}", 
                                       vehicle_class, 
                                       count, 
                                       f"{percentage:.2f}%"])
                    
                    # Add separator between minutes
                    writer.writerow([])
                
                # Write current minute if not empty
                total_current = sum(len(ids) for ids in self.current_minute_counts.values())
                if total_current > 0:
                    for vehicle_class, ids in self.current_minute_counts.items():
                        count = len(ids)
                        percentage = (count / total_current * 100) if total_current > 0 else 0
                        writer.writerow([f"Minute {self.current_minute + 1}", 
                                       vehicle_class, 
                                       count, 
                                       f"{percentage:.2f}%"])
            
            print(f"\nResults saved to: {self.csv_filename}")
            
        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")

    def print_statistics(self):
        """Print final statistics"""
        print("\nVehicle Detection Statistics by Minute")
        print("=====================================")
        
        for minute, counts in enumerate(self.minute_counts):
            total = sum(len(ids) for ids in counts.values())
            print(f"\nMinute {minute + 1}:")
            print(f"Total Vehicles: {total}")
            print("Vehicle Class Distribution:")
            
            for vehicle_class, ids in counts.items():
                count = len(ids)
                percentage = (count / total * 100) if total > 0 else 0
                print(f"{vehicle_class}: {count} ({percentage:.1f}%)")
        
        # Print current minute if not empty
        total_current = sum(len(ids) for ids in self.current_minute_counts.values())
        if total_current > 0:
            print(f"\nCurrent Minute {self.current_minute + 1}:")
            print(f"Total Vehicles: {total_current}")
            print("Vehicle Class Distribution:")
            
            for vehicle_class, ids in self.current_minute_counts.items():
                count = len(ids)
                percentage = (count / total_current * 100) if total_current > 0 else 0
                print(f"{vehicle_class}: {count} ({percentage:.1f}%)")