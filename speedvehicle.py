import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from tracker import Sort
import csv
from datetime import datetime
import mysql.connector
from mysql.connector import Error

# Add DB configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'Traffic1',
    'password': 'pedestrian',
    'database': 'traffic_db'
}

class VehicleSpeedDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the Vehicle Speed Detector
        
        Args:
            model_path (str): Path to the YOLOv8 model weights
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Dictionary to store vehicle tracks
        self.vehicle_tracks = {}
        
        # Initialize tracker
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Store previous frame information for speed calculation
        self.prev_tracks = {}
        self.speed_estimates = {}
        
        # Camera parameters (to be calibrated)
        self.pixels_per_meter = None  # Will be set during calibration
        self.fps = None
        
        # Modified variables for multiple lines
        self.base_line = None
        self.reference_lines = []
        self.line_distances = [0, 5, 10, 15]  # Distances in meters from base line
        self.crossing_times = {}  # Format: {track_id: {line_idx: time}}
        self.speeds = {}  # Format: {track_id: {line_idx: speed}}
        self.avg_speeds = {}  # Format: {track_id: avg_speed}
        
        # Add database connection
        self.video_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            self.db_connection = mysql.connector.connect(**DB_CONFIG)
            self.db_cursor = self.db_connection.cursor()
            print("Successfully connected to MySQL database")
            
            # Setup static data for this video session
            self.setup_sample_data()
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            self.db_connection = None
            self.db_cursor = None
        
    def set_camera_parameters(self, pixels_per_meter: float, fps: float):
        """Set camera calibration parameters"""
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        
    def select_reference_line(self, frame):
        """Allow user to select single reference line"""
        points = []
        frame_copy = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) == 2:  # Complete line
                    cv2.line(frame_copy, points[0], points[1], (0, 0, 255), 2)
                    cv2.imshow('Select Base Reference Line', frame_copy)
                else:  # First point
                    cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow('Select Base Reference Line', frame_copy)

        cv2.namedWindow('Select Base Reference Line')
        cv2.setMouseCallback('Select Base Reference Line', mouse_callback)
        
        print("Select base reference line (2 points). Press 'Enter' when done.")
        while True:
            cv2.imshow('Select Base Reference Line', frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points) == 2:  # Enter key and line selected
                break
        
        cv2.destroyWindow('Select Base Reference Line')
        self.base_line = (points[0], points[1])
        # Note: We'll generate reference lines after setting camera parameters

    def _generate_reference_lines(self):
        """Generate additional reference lines above base line"""
        if not self.base_line:
            return
        
        p1, p2 = self.base_line
        line_vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        perpendicular = np.array([-line_vector[1], line_vector[0]])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        self.reference_lines = []
        for distance in self.line_distances:
            offset = distance * self.pixels_per_meter
            offset_vector = perpendicular * offset
            new_p1 = (int(p1[0] + offset_vector[0]), int(p1[1] + offset_vector[1]))
            new_p2 = (int(p2[0] + offset_vector[0]), int(p2[1] + offset_vector[1]))
            self.reference_lines.append((new_p1, new_p2))

    def setup_sample_data(self):
        """Setup static data for the video session"""
        try:
            # Static data for the entire video
            queries = [
                """INSERT IGNORE INTO location_dimension 
                   (location_key, metro_city_province, district, neighborhood, spot) 
                   VALUES (1, 'Sample City', 'Sample District', 'Sample Area', 'Sample Spot')""",
                
                """INSERT IGNORE INTO road_character_dimension 
                   (road_key, road_type, road_feature) 
                   VALUES (1, 'Sample Road Type', 'Sample Feature')""",
                
                """INSERT IGNORE INTO behavior_feature 
                   (behavior_id, behavior_feature) 
                   VALUES 
                   (1, 'L1_speed'),
                   (2, 'L2_speed'),
                   (3, 'L3_speed'),
                   (4, 'avg_speed')"""
            ]
            
            for query in queries:
                self.db_cursor.execute(query)
            self.db_connection.commit()
            print("Static video data setup complete")
        except Error as e:
            print(f"Error setting up static data: {e}")

    def insert_speed_data(self, track_id: int, speeds: dict, avg_speed: float, timestamp: float):
        """Insert speed data into database"""
        if not hasattr(self, 'db_cursor') or self.db_cursor is None:
            print("Database connection not available")
            return
        
        try:
            if not self.db_connection.is_connected():
                print("Reconnecting to database...")
                self.db_connection.reconnect()
                self.db_cursor = self.db_cursor()

            # Check if this ID already exists
            check_query = "SELECT time_key FROM time_dimension WHERE time_key = %s"
            self.db_cursor.execute(check_query, (track_id,))
            if self.db_cursor.fetchone():
                print(f"ID {track_id} already exists in database, skipping...")
                return

            current_time = datetime.now()
            
            # Insert time dimension
            time_query = """
            INSERT INTO time_dimension 
            (time_key, week, day, day_night, hour) 
            VALUES (%s, %s, %s, %s, %s)
            """
            time_key = track_id
            week = f"Week{current_time.strftime('%V')}"
            day = current_time.strftime('%A')
            day_night = 'Day' if 6 <= current_time.hour <= 18 else 'Night'
            hour = current_time.strftime('%Y-%m-%d')
            
            self.db_cursor.execute(time_query, (time_key, week, day, day_night, hour))

            # Insert scene dimension
            scene_query = """
            INSERT INTO scene_dimension 
            (scene_key, object_type, event_type) 
            VALUES (%s, %s, %s)
            """
            self.db_cursor.execute(scene_query, (track_id, 'vehicle', 'movement'))

            # Insert fact table
            fact_query = """
            INSERT INTO fact_table 
            (time_key, location_key, road_key, scene_key, scene_ratio, video_code) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.db_cursor.execute(fact_query, (
                time_key, 1, 1, track_id, 1.0, f"video_{self.video_id}_{track_id}"
            ))

            # Insert speed data for each line and average
            behavior_query = """
            INSERT INTO scene_behavior_feature 
            (scene_key, object_type, behavior_id, behavior_value) 
            VALUES (%s, %s, %s, %s)
            """
            
            # Insert individual line speeds
            for line_idx, speed in speeds.items():
                behavior_id = line_idx  # behavior_id 1,2,3 for L1,L2,L3
                self.db_cursor.execute(behavior_query, (
                    track_id, 'vehicle', behavior_id, speed
                ))
            
            # Insert average speed
            self.db_cursor.execute(behavior_query, (
                track_id, 'vehicle', 4, avg_speed  # behavior_id 4 for average speed
            ))

            self.db_connection.commit()
            print(f"Successfully inserted speed data for vehicle {track_id}")
            
        except Error as e:
            print(f"Error inserting speed data: {str(e)}")
            self.db_connection.rollback()

    def cleanup(self):
        """Close database connection"""
        if hasattr(self, 'db_cursor') and self.db_cursor:
            self.db_cursor.close()
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for vehicle detection and speed estimation
        
        Args:
            frame (np.ndarray): Input frame
            frame_id (int): Frame number
            
        Returns:
            Tuple[np.ndarray, Dict]: Processed frame with annotations and detection results
        """
        try:
            # Run YOLOv8 inference
            results = self.model(frame)[0]
            
            # Process detections (only class_id 4)
            detections = []
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.conf_threshold and int(cls) == 4:  # Only process class_id 4
                    detections.append([x1, y1, x2, y2, conf])
            
            # Update tracker
            if len(detections) > 0:
                tracks = self.tracker.update(np.array(detections))
            else:
                tracks = np.empty((0, 5))
                
            # Calculate current time in seconds
            current_time = frame_id / self.fps
            
            # Process each track
            current_tracks = {}
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                current_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                current_tracks[track_id] = current_center
                
                # Initialize data structures for new tracks
                if track_id not in self.crossing_times:
                    self.crossing_times[track_id] = {}
                    self.speeds[track_id] = {}
                
                # Check crossings for all lines
                if track_id in self.prev_tracks:
                    prev_center = self.prev_tracks[track_id]
                    for line_idx, line in enumerate(self.reference_lines):
                        if line_idx not in self.crossing_times[track_id]:
                            if self._check_line_crossing(prev_center, current_center, line):
                                self.crossing_times[track_id][line_idx] = current_time
                                
                                # Calculate speed if it's not the first line
                                if line_idx > 0:
                                    prev_crossing = self.crossing_times[track_id][line_idx - 1]
                                    time_diff = current_time - prev_crossing
                                    if time_diff > 0:
                                        distance = 5  # 5m between lines
                                        speed = (distance / time_diff) * 3.6  # Convert to km/h
                                        self.speeds[track_id][line_idx] = speed
                                        
                                        # Calculate average speed if vehicle has crossed all lines
                                        if len(self.speeds[track_id]) == len(self.reference_lines) - 1:
                                            avg_speed = sum(self.speeds[track_id].values()) / len(self.speeds[track_id])
                                            self.avg_speeds[track_id] = avg_speed
                                            # Insert into database instead of CSV
                                            self.insert_speed_data(
                                                track_id,
                                                self.speeds[track_id],
                                                avg_speed,
                                                current_time
                                            )
            
            # Update previous tracks
            self.prev_tracks = current_tracks
            
            # Draw reference lines and tracks on frame
            frame = self._draw_reference_lines(frame.copy())
            annotated_frame = self._draw_tracks(frame, tracks)
            
            return annotated_frame, {
                'tracks': tracks.tolist() if len(tracks) > 0 else [],
                'speeds': self.speeds,
                'avg_speeds': self.avg_speeds
            }
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            return frame, {'tracks': [], 'speeds': {}, 'avg_speeds': {}}
    
    def _draw_tracks(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """Draw tracking boxes and speed information on the frame"""
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw ID and speeds
            text_lines = [f"ID: {track_id}"]
            if track_id in self.speeds:
                for line_idx, speed in self.speeds[track_id].items():
                    text_lines.append(f"L{line_idx}: {speed:.1f}")
            if track_id in self.avg_speeds:
                text_lines.append(f"Avg: {self.avg_speeds[track_id]:.1f}")
            
            # Draw text
            y_offset = int(y1) - 10
            for text in text_lines:
                cv2.putText(frame, text, (int(x1), y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset -= 20
                       
        return frame

    def _draw_reference_lines(self, frame):
        """Draw all reference lines on frame"""
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Different color for each line
        for idx, line in enumerate(self.reference_lines):
            cv2.line(frame, line[0], line[1], colors[idx], 2)
            cv2.putText(frame, f"Line {idx}", 
                       (line[0][0], line[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[idx], 2)
        return frame

    def _check_line_crossing(self, prev_center, current_center, line):
        """Check if a track crossed the line"""
        line_p1, line_p2 = line
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Return true if line segments intersect
        return ccw(prev_center, line_p1, line_p2) != ccw(current_center, line_p1, line_p2) and \
               ccw(prev_center, current_center, line_p1) != ccw(prev_center, current_center, line_p2)

def main():
    parser = argparse.ArgumentParser(description='Vehicle Speed Detection using YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to input video file')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video')
    parser.add_argument('--pixels-per-meter', type=float, required=True,
                       help='Pixels per meter ratio for speed calculation')
    parser.add_argument('--display', action='store_true', help='Display output in real-time')
    parser.add_argument('--distance', type=float, required=True,
                       help='Real-world distance between reference lines in meters')
    parser.add_argument('--csv-output', type=str, default='speeds.csv',
                       help='Path to output CSV file')
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Check if video file exists
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Video file not found: {args.source}")
    
    # Initialize detector
    detector = VehicleSpeedDetector(args.model, args.conf_thres)
    
    # Open video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {args.source}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Read first frame for reference line selection
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    # Select reference line
    detector.select_reference_line(first_frame)
    
    # Update detector parameters
    detector.set_camera_parameters(args.pixels_per_meter, fps)
    
    # Now generate reference lines after parameters are set
    detector._generate_reference_lines()
    
    # Reset video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize video writer
    output_path = Path(args.output)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    frame_count = 0
    processed_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Process frame
                processed_frame, results = detector.process_frame(frame, frame_count)
                
                # Write frame
                writer.write(processed_frame)
                
                # Display frame if requested
                if args.display:
                    # Add frame counter and FPS to display
                    cv2.putText(
                        processed_frame,
                        f"Frame: {frame_count} | Tracked Objects: {len(results['tracks'])}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.imshow('Vehicle Speed Detection', processed_frame)
                    
                    # Break if 'q' is pressed or window is closed
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break
                
                processed_count += 1
                
                # Display progress
                if frame_count % 30 == 0:
                    print(f"Processed frame {frame_count}")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                writer.write(frame)  # Write original frame if processing fails
                
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        detector.cleanup()  # Add cleanup call
        print(f"\nProcessed {processed_count} frames out of {frame_count} total frames")
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main() 

    
