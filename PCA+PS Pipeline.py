import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import os
import time
from tracker import Sort
import csv
from datetime import datetime
from typing import Dict, List, Tuple
from PCA import PedestrianCrossingAnalyzer
from PS import PedestrianSpeedDetector

class PedestrianPipeline:
    def __init__(self, model_path: str, pixels_per_meter: float, conf_threshold: float = 0.5):
        """Initialize the main pipeline"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Initialize YOLO model
            self.model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")
            
            # Initialize tracker
            self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
            
            # Initialize external processors
            self.crossing_analyzer = PedestrianCrossingAnalyzer()
            self.speed_detector = PedestrianSpeedDetector(pixels_per_meter)
            
            
            # Setup CSV logging
            self.setup_csv_logging()
            
            # Store results
            self.results = {}
            
            print("Pipeline initialization complete.")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def setup_csv_logging(self):
        """Setup CSV file for logging all results"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = f'logs/pedestrian_analysis_{timestamp}.csv'
        
        # Create headers for all data
        headers = ['Pedestrian_ID']
        # Add crossing time headers
        for lane in range(3):
            for line in range(3):
                headers.append(f'Lane{lane+1}_Line{line+1}_Time')
        # Add speed headers
        headers.extend(['Average_Speed', 'Max_Speed'])
        
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        self.csv_rows = {}

    def process_video(self, video_path: str):
        """Process video file and coordinate external processing"""
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        try:
            print(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Read first frame for point selection
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")
            
            # First run PCA point selection
            print("\nStarting point selection for crossing analysis...")
            self.crossing_analyzer.select_points(first_frame)
            print("Point selection complete. Starting main video processing...")
            
            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create output directory
            os.makedirs('output', exist_ok=True)
            
            # Create output video writer
            output_path = os.path.join('output', f"processed_{os.path.basename(video_path)}")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get current time in HH:MM:SS format
                current_time = datetime.now().strftime('%H:%M:%S')
                frame_time = frame_count / fps
                
                # Process detections
                results = self.model(frame)[0]
                detections = []
                
                for det in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = det
                    
                    # Filter detections
                    min_width = 20
                    min_height = 40
                    width = x2 - x1
                    height = y2 - y1
                    
                    if (int(cls) == 3 and  # Pedestrian class
                        conf > 0.5 and     # Confidence threshold
                        width > min_width and 
                        height > min_height):
                        detections.append([x1, y1, x2, y2, 1.0])
                
                # Update tracker
                if len(detections) > 0:
                    tracks = self.tracker.update(np.array(detections))
                else:
                    tracks = np.empty((0, 5))
                
                # Process tracks with external modules
                for track in tracks:
                    track_id = int(track[4])
                    bbox = track[:4]
                    mid_x = int((bbox[0] + bbox[2]) / 2)
                    mid_y = int((bbox[1] + bbox[3]) / 2)
                    
                    # Process speed
                    speed_results = self.speed_detector.process_track(track_id, (mid_x, mid_y), frame_time)
                    
                    # Process crossing
                    crossing_results = self.crossing_analyzer.process_track(track_id, (mid_x, mid_y), frame_time)
                    
                    # Combine results
                    if track_id not in self.results:
                        self.results[track_id] = {
                            'crossing_times': {},
                            'speeds': []
                        }
                    
                    # Update results
                    self.results[track_id]['crossing_times'].update(crossing_results.get('crossing_times', {}))
                    self.results[track_id]['speeds'].extend(speed_results.get('speeds', []))
                    
                    # Update CSV if new data
                    if crossing_results.get('crossing_times') or speed_results.get('speeds'):
                        self.update_csv(track_id)
                    
                    # Draw visualization
                    self.draw_visualization(frame, track_id, bbox, mid_x, mid_y, 
                                         speed_results.get('current_speed', 0))
                
                # Draw crossing lines
                self.crossing_analyzer.draw_lines(frame)
                
                # Show current time
                cv2.putText(frame, f"Time: {current_time}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                out.write(frame)
                cv2.imshow('Pedestrian Analysis', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
            # Write final results
            self.write_final_results()
            print(f"\nProcessing complete! Output saved to: {output_path}")
            
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            raise
            
        finally:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()

    def draw_visualization(self, frame, track_id, bbox, mid_x, mid_y, current_speed):
        """Draw visualization elements on frame"""
        color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), color, 2)
        
        # Draw ID and speed
        cv2.putText(frame, f"ID:{track_id}", (int(bbox[0]), int(bbox[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"{current_speed:.1f} m/s", (int(bbox[0]), int(bbox[3])+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw center point
        cv2.circle(frame, (mid_x, mid_y), 4, (0, 0, 255), -1)

    def update_csv(self, track_id):
        """Update CSV with new data for a track"""
        if track_id in self.results:
            track_data = self.results[track_id]
            crossing_times = track_data['crossing_times']
            speeds = track_data['speeds']
            
            # Only process if we have speeds
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
                
                # Get crossing times in order
                times = []
                for line in range(9):
                    time_str = crossing_times.get(line, "")
                    times.append(time_str)
                
                # Update CSV row
                self.csv_rows[track_id] = [track_id] + times + [f"{avg_speed:.2f}", f"{max_speed:.2f}"]
                
                # Write to CSV if we have all crossing times
                if len(crossing_times) == 9 and all(times):
                    with open(self.csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        # Write header
                        headers = ['Pedestrian_ID']
                        for lane in range(3):
                            for line in range(3):
                                headers.append(f'Lane{lane+1}_Line{line+1}_Time')
                        headers.extend(['Average_Speed', 'Max_Speed'])
                        writer.writerow(headers)
                        
                        # Write all complete rows
                        for id, row_data in self.csv_rows.items():
                            writer.writerow(row_data)

    def write_final_results(self):
        """Write final results to CSV"""
        # Update any remaining unwritten tracks
        for track_id in self.results:
            track_data = self.results[track_id]
            crossing_times = track_data['crossing_times']
            speeds = track_data['speeds']
            
            # Only process if we have speeds
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
                
                # Get crossing times in order
                times = []
                for line in range(9):
                    time_str = crossing_times.get(line, "")
                    times.append(time_str)
                
                # Only add if we have all crossing times
                if len(crossing_times) == 9 and all(times):
                    self.csv_rows[track_id] = [track_id] + times + [f"{avg_speed:.2f}", f"{max_speed:.2f}"]
        
        # Write final CSV with all complete records
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Pedestrian_ID']
            for lane in range(3):
                for line in range(3):
                    headers.append(f'Lane{lane+1}_Line{line+1}_Time')
            headers.extend(['Average_Speed', 'Max_Speed'])
            writer.writerow(headers)
            
            # Write only complete records
            for id, row_data in self.csv_rows.items():
                if len(row_data) == 12:  # ID + 9 times + 2 speeds
                    writer.writerow(row_data)

    def process_track(self, track_id: int, position: Tuple[float, float], frame_time: float) -> Dict:
        """Process a track and return crossing information"""
        # Initialize crossing times for new track
        if track_id not in self.crossing_times:
            self.crossing_times[track_id] = {}
        
        # Check line crossings
        line_crossed = self.check_line_crossing(position, frame_time)
        if line_crossed is not None:
            # Get current time in HH:MM:SS format
            current_time = datetime.now().strftime('%H:%M:%S')
            # Record the time for this line crossing
            self.crossing_times[track_id][line_crossed] = current_time
            print(f"Pedestrian {track_id} crossed line {line_crossed+1} at {current_time}")
        
        return {
            'crossing_times': self.crossing_times.get(track_id, {})
        }

def main():
    try:
        # Configuration
        model_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\best(3).pt"
        video_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\export6.mp4"
        pixels_per_meter = 100  # Adjust this value based on your video's scale
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        
        pipeline = PedestrianPipeline(model_path, pixels_per_meter)
        pipeline.process_video(video_path)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return

if __name__ == "__main__":
    main()
