import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from tracker import Sort
import math
import mysql.connector
from mysql.connector import Error
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'user': 'Traffic1',
    'password': 'pedestrian',
    'database': 'traffic_db'
}

class PedestrianSpeedDetector:
    def __init__(self, model_path: str, pixels_per_meter: float, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be positive")
        self.pixels_per_meter = float(pixels_per_meter)
        print(f"Conversion factor: {self.pixels_per_meter} pixels/meter")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Store pedestrian tracking data
        self.pedestrian_tracks = {}
        self.fps = None
        self.colors = np.random.randint(0, 255, size=(100, 3))
        self.speed_data = {}
        self.video_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.processed_ids = set()
        
        # Add trajectory storage
        self.trajectories = {}  # {track_id: [(x1,y1), (x2,y2), ...]}
        self.trajectory_colors = {}  # {track_id: color}
        
        # Initialize database connection
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
                   VALUES (1, 'speed')"""
            ]
            
            for query in queries:
                self.db_cursor.execute(query)
            self.db_connection.commit()
            print("Static video data setup complete")
        except Error as e:
            print(f"Error setting up static data: {e}")

    def insert_transformed_speed(self, track_id: int, speed: float):
        """Insert speed data with duplicate handling"""
        if not hasattr(self, 'db_cursor') or self.db_cursor is None:
            print("Database connection not available")
            return
        
        try:
            if not self.db_connection.is_connected():
                print("Reconnecting to database...")
                self.db_connection.reconnect()
                self.db_cursor = self.db_connection.cursor()

            # First check if this ID already exists
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
            self.db_cursor.execute(scene_query, (track_id, 'pedestrian', 'movement'))

            # Insert fact table
            fact_query = """
            INSERT INTO fact_table 
            (time_key, location_key, road_key, scene_key, scene_ratio, video_code) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.db_cursor.execute(fact_query, (
                time_key, 1, 1, track_id, 1.0, f"video_{self.video_id}_{track_id}"
            ))

            # Insert speed data
            behavior_query = """
            INSERT INTO scene_behavior_feature 
            (scene_key, object_type, behavior_id, behavior_value) 
            VALUES (%s, %s, %s, %s)
            """
            self.db_cursor.execute(behavior_query, (
                track_id, 'pedestrian', 1, speed
            ))

            self.db_connection.commit()
            print(f"Successfully inserted speed data for pedestrian {track_id}: {speed:.2f} m/s")
            
        except Error as e:
            print(f"Error inserting speed data: {str(e)}")
            self.db_connection.rollback()

    def calculate_speed(self, positions: List[Tuple[float, float]], times: List[float]) -> float:
        """Calculate speed in meters per second using last N positions"""
        try:
            if len(positions) < 2 or len(times) < 2:
                print(f"Not enough data points: positions={len(positions)}, times={len(times)}")
                return 0.0
            
            # Use last 5 positions or all if less
            n_positions = min(5, len(positions))
            recent_positions = positions[-n_positions:]
            recent_times = times[-n_positions:]
            
            # Calculate total distance in pixels
            total_distance = 0
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                distance = math.sqrt(dx*dx + dy*dy)
                total_distance += distance
            
            # Convert distance to meters
            distance_meters = total_distance / self.pixels_per_meter
            
            # Calculate time difference
            time_diff = recent_times[-1] - recent_times[0]
            
            if time_diff > 0:
                speed = distance_meters / time_diff
                print(f"Speed calculation: distance={distance_meters:.2f}m, time={time_diff:.2f}s, speed={speed:.2f}m/s")
                return speed
            else:
                print(f"Invalid time difference: {time_diff}")
                return 0.0
            
        except Exception as e:
            print(f"Error in speed calculation: {e}")
            return 0.0

    def draw_trajectories(self, frame):
        """Draw all trajectories on the frame"""
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
                
            # Get color for this track
            if track_id not in self.trajectory_colors:
                self.trajectory_colors[track_id] = self.colors[track_id % len(self.colors)].tolist()
            color = self.trajectory_colors[track_id]
            
            # Draw trajectory line
            points = np.array(trajectory, dtype=np.int32)
            cv2.polylines(frame, [points], False, color, 2)
            
            # Draw speed if available
            if track_id in self.pedestrian_tracks and self.pedestrian_tracks[track_id]['speeds']:
                speed = self.pedestrian_tracks[track_id]['speeds'][-1]
                last_point = trajectory[-1]
                cv2.putText(frame,
                           f"{speed:.2f} m/s",
                           (last_point[0] + 10, last_point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def process_frame(self, frame: np.ndarray, frame_id: int, tracker: Sort) -> Tuple[np.ndarray, Dict]:
        try:
            current_time = frame_id / self.fps if self.fps else 0
            results = self.model(frame)[0]
            
            # Process detections
            detections = []
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.conf_threshold and int(cls) == 3:  # class 3 is person
                    detections.append([x1, y1, x2, y2, conf])
            
            # Update tracker with current detections
            if detections:
                tracks = tracker.update(np.array(detections))
            else:
                tracks = np.empty((0, 5))
            
            # Process each track
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int(bbox[3])  # Use bottom center
                center = (center_x, center_y)
                
                # Update trajectory
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = []
                self.trajectories[track_id].append(center)
                
                # Limit trajectory length
                max_trajectory_length = 50
                if len(self.trajectories[track_id]) > max_trajectory_length:
                    self.trajectories[track_id] = self.trajectories[track_id][-max_trajectory_length:]
                
                # Skip if already processed
                if track_id in self.processed_ids:
                    continue
                
                # Initialize or update track data
                if track_id not in self.pedestrian_tracks:
                    self.pedestrian_tracks[track_id] = {
                        'positions': [],
                        'speeds': [],
                        'times': [],
                        'last_update': current_time
                    }
                
                track_data = self.pedestrian_tracks[track_id]
                
                # Update position data
                if current_time - track_data['last_update'] > 0.1:  # Update every 0.1 seconds
                    track_data['positions'].append(center)
                    track_data['times'].append(current_time)
                    track_data['last_update'] = current_time
                
                    # Calculate speed if we have enough data points
                    if len(track_data['positions']) >= 5:
                        speed = self.calculate_speed(
                            track_data['positions'][-5:],
                            track_data['times'][-5:]
                        )
                        
                        if speed > 0 and speed < 10:  # reasonable speed range
                            track_data['speeds'].append(speed)
                            
                            # Process if we have consistent tracking
                            if len(track_data['speeds']) >= 3:
                                avg_speed = np.mean(track_data['speeds'][-3:])
                                if track_id not in self.processed_ids:
                                    self.insert_transformed_speed(track_id, avg_speed)
                                    self.processed_ids.add(track_id)
                                    print(f"Processed pedestrian {track_id}: {avg_speed:.2f} m/s")
                
                # Visualization
                color = self.colors[track_id % len(self.colors)]
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                
                status = "PROCESSED" if track_id in self.processed_ids else "TRACKING"
                if track_id in self.pedestrian_tracks and self.pedestrian_tracks[track_id]['speeds']:
                    speed = self.pedestrian_tracks[track_id]['speeds'][-1]
                    status = f"{status} {speed:.2f}m/s"
                
                cv2.putText(frame, 
                           f"ID:{track_id} ({status})",
                           (center_x, center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw all trajectories
            frame = self.draw_trajectories(frame)
            
            return frame, {
                'tracks': tracks.tolist() if len(tracks) > 0 else [],
                'pedestrian_tracks': self.pedestrian_tracks
            }
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            return frame, {'tracks': [], 'pedestrian_tracks': {}}

    def cleanup(self):
        """Close database connection"""
        if hasattr(self, 'db_cursor') and self.db_cursor:
            self.db_cursor.close()
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()

def main():
    parser = argparse.ArgumentParser(description='Pedestrian Speed Detection using YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to input video file')
    parser.add_argument('--pixels-per-meter', type=float, required=True, help='Pixels per meter conversion factor')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--display', action='store_true', help='Enable display window')
    args = parser.parse_args()
    
    # Initialize detector and tracker
    detector = PedestrianSpeedDetector(
        args.model, 
        args.pixels_per_meter, 
        args.conf_thres
    )
    
    tracker = Sort(max_age=15, min_hits=5, iou_threshold=0.5)
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {args.source}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    detector.fps = fps if fps > 0 else 30
    
    frame_count = 0
    try:
        # Get first frame to set up display window
        ret, frame = cap.read()
        if ret:
            # Create window with a reasonable size
            cv2.namedWindow('Pedestrian Tracking', cv2.WINDOW_NORMAL)
            height, width = frame.shape[:2]
            cv2.resizeWindow('Pedestrian Tracking', width, height)
        
        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"\nProcessing frame {frame_count + 1}")
            processed_frame, results = detector.process_frame(frame, frame_count, tracker)
            
            if args.display:
                cv2.imshow('Pedestrian Tracking', processed_frame)
                # Add a longer wait time to make the display more visible
                key = cv2.waitKey(30) & 0xFF  # Changed from 1 to 30
                if key == ord('q'):
                    print("\nStopped by user")
                    break
            
            frame_count += 1
        
        print("\n=== Final Summary ===")
        print(f"Total unique pedestrians processed: {len(detector.processed_ids)}")
        print("\nProcessed Pedestrian IDs and their speeds:")
        for ped_id in detector.processed_ids:
            track_data = detector.pedestrian_tracks[ped_id]
            if track_data['speeds']:
                speed = track_data['speeds'][-1]
                print(f"Pedestrian {ped_id}: {speed:.2f} m/s ({speed * 3.6:.2f} km/h)")
        
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        print("\nProgram terminated")

if __name__ == "__main__":
    main() 