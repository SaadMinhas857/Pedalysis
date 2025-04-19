import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from tracker import Sort
import math
from datetime import datetime
import csv
import os

class PedestrianSpeedDetector:
    def __init__(self, pixels_per_meter: float, csv_filename: str):
        """Initialize Speed Detector"""
        self.pixels_per_meter = pixels_per_meter
        self.track_history = {}  # {track_id: [(x, y, time), ...]}
        self.speeds = {}  # {track_id: [speed1, speed2, ...]}
        
        # Store CSV filename
        self.csv_filename = csv_filename
        
        # Setup CSV file
        self.setup_csv_file()
        
        print("Speed Detector initialized.")

    def setup_csv_file(self):
        """Setup CSV file with headers"""
        os.makedirs('logs', exist_ok=True)
        headers = ['Pedestrian_ID']
        for lane in range(3):
            for line in range(3):
                headers.append(f'Lane{lane+1}_Line{line+1}_Time')
        headers.extend(['Average_Speed', 'Max_Speed'])
        
        try:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            print(f"Error setting up CSV file: {str(e)}")
            raise

    def process_frame(self, frame, tracks, frame_time):
        """Process a frame with tracked objects"""
        processed_frame = frame.copy()
        
        # Process each track
        for track in tracks:
            track_id = int(track[4])
            bbox = track[:4]
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            
            # Record position and time
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((center, frame_time))
            
            # Calculate speed if we have enough history
            if len(self.track_history[track_id]) >= 2:
                # Get positions and times
                pos1, time1 = self.track_history[track_id][-2]
                pos2, time2 = self.track_history[track_id][-1]
                
                # Calculate distance in pixels
                distance_px = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                
                # Convert to meters
                distance_m = distance_px / self.pixels_per_meter
                
                # Calculate time difference
                time_diff = time2 - time1
                
                if time_diff > 0:
                    # Calculate speed in km/h
                    speed = (distance_m / time_diff) * 3.6
                    
                    # Store speed in track history
                    if track_id not in self.speeds:
                        self.speeds[track_id] = []
                    self.speeds[track_id].append(speed)
                    
                    # Calculate average speed
                    avg_speed = sum(self.speeds[track_id]) / len(self.speeds[track_id])
                    
                    # Store in database
                    speed_data = {
                        12: avg_speed  # Behavior ID 12 is for pedestrian_speed
                    }
                    self.pipeline.insert_track_data(track_id, 'pedestrian', speed_data)
                    
                    # Draw speed on frame
                    cv2.putText(processed_frame,
                              f"Speed: {avg_speed:.1f} km/h",
                              (int(bbox[0]), int(bbox[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.7,
                              (0, 255, 0),
                              2)
            
            # Draw bounding box and ID
            cv2.rectangle(processed_frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        (0, 255, 0),
                        2)
            
            cv2.putText(processed_frame,
                       f"ID: {track_id}",
                       (int(bbox[0]), int(bbox[1]) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)
        
        return processed_frame

    def calculate_speed(self, track_id):
        """Calculate speed for a track"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return 0.0
        
        history = self.track_history[track_id]
        # Use last two positions for speed calculation
        (x1, y1, t1), (x2, y2, t2) = history[-2:]
        
        # Calculate distance in pixels
        distance_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Convert to meters
        distance_meters = distance_pixels / self.pixels_per_meter
        
        # Calculate time difference in seconds
        time_diff = t2 - t1
        
        # Calculate speed in m/s
        speed = distance_meters / time_diff if time_diff > 0 else 0.0
        
        return speed

    def write_speed_to_csv(self, track_id):
        """Write speed data to CSV"""
        if track_id in self.speeds and len(self.speeds[track_id]) >= 10:
            speeds = self.speeds[track_id]
            avg_speed = float(sum(speeds)) / float(len(speeds))
            max_speed = float(max(speeds))
            
            # Read existing data
            rows = []
            headers = ['Pedestrian_ID']
            for lane in range(3):
                for line in range(3):
                    headers.append(f'Lane{lane+1}_Line{line+1}_Time')
            headers.extend(['Average_Speed', 'Max_Speed'])
            
            try:
                with open(self.csv_filename, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            except FileNotFoundError:
                pass
            
            # Update or add row for this track
            updated = False
            for row in rows:
                if row['Pedestrian_ID'] == str(track_id):  # Compare with string ID
                    row['Average_Speed'] = f"{avg_speed:.2f}"
                    row['Max_Speed'] = f"{max_speed:.2f}"
                    updated = True
                    break
            
            if not updated:
                new_row = {'Pedestrian_ID': str(track_id)}
                # Initialize all time columns to empty
                for lane in range(3):
                    for line in range(3):
                        new_row[f'Lane{lane+1}_Line{line+1}_Time'] = ''
                new_row['Average_Speed'] = f"{avg_speed:.2f}"
                new_row['Max_Speed'] = f"{max_speed:.2f}"
                rows.append(new_row)
            
            # Write back all data
            try:
                with open(self.csv_filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
            except Exception as e:
                print(f"Error writing to CSV: {str(e)}")

def main():
    try:
        # Configuration
        video_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\export6.mp4"
        pixels_per_meter = 100  # Adjust this value based on your video's scale
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        
        # Initialize detector
        detector = PedestrianSpeedDetector(pixels_per_meter, 'logs/pedestrian_speeds.csv')
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        os.makedirs('output', exist_ok=True)
        output_path = os.path.join('output', f"speed_{os.path.basename(video_path)}")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = detector.process_frame(frame, [], frame_count/fps)
                
                # Write frame
                writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('Pedestrian Speed Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            
            print(f"\nProcessed {frame_count} frames")
            print(f"Video output saved to: {output_path}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return

if __name__ == "__main__":
    main() 