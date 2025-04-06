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
            
            # Class mapping for your custom model
            self.class_mapping = {
                0: 'biker',
                1: 'bus',      # Bus class
                2: 'motobike', # Motobike class
                3: 'pedestrian',
                4: 'sedan',    # Sedan class
                5: 'taxi',     # Taxi class
                6: 'truck'
            }
            
            # Setup CSV file
            self.setup_csv_file()
            
            print("VehicleAnalyzer initialization complete.")
            
        except Exception as e:
            raise Exception(f"Initialization error: {str(e)}")

    def setup_csv_file(self):
        """Setup CSV file with headers"""
        os.makedirs('logs', exist_ok=True)
        headers = ['Vehicle Class', 'Count', 'Percentage']
        
        try:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            print(f"Error setting up CSV file: {str(e)}")
            raise

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

    def process_frame(self, frame: np.ndarray, tracks: np.ndarray, vehicle_classes: Dict[int, int]) -> np.ndarray:
        """Process a frame with tracked objects and their classes"""
        try:
            processed_frame = frame.copy()
            frame_counts = {class_name: 0 for class_name in self.vehicle_counts.keys()}
            
            # Process tracked objects
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                
                # Get vehicle class from pipeline's vehicle_classes
                if track_id in vehicle_classes:
                    cls = vehicle_classes[track_id]
                    vehicle_class = self.class_mapping[cls]
                    
                    # Update unique vehicles count
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
            return processed_frame
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame

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

    def save_results_to_csv(self):
        """Save results to CSV file"""
        try:
            # Prepare data for CSV
            total_vehicles = sum(len(ids) for ids in self.vehicle_counts.values())
            
            # Write data
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Vehicle Class', 'Count', 'Percentage'])
                
                # Write vehicle counts
                for vehicle_class, ids in self.vehicle_counts.items():
                    count = len(ids)
                    percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
                    writer.writerow([vehicle_class, count, f"{percentage:.2f}%"])
                
                # Add total row
                writer.writerow(['Total', total_vehicles, '100.00%'])
            
            print(f"\nResults saved to: {self.csv_filename}")
            
        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")

    def print_statistics(self):
        """Print final statistics"""
        total_vehicles = sum(len(ids) for ids in self.vehicle_counts.values())
        
        print("\nVehicle Detection Statistics")
        print("==========================")
        print(f"Total Unique Vehicles: {total_vehicles}")
        print("\nVehicle Class Distribution:")
        
        for vehicle_class, ids in self.vehicle_counts.items():
            count = len(ids)
            percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
            print(f"{vehicle_class}: {count} ({percentage:.1f}%)")