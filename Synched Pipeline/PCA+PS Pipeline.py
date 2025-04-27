import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime, timedelta
from PCA import PedestrianCrossingAnalyzer
from PS import PedestrianSpeedDetector
from CARPSM import CarPSMDetector
from VCOMP import VehicleAnalyzer
from UltralyticsProcessor import UltralyticsProcessor
import csv
from tracker import Sort
import supervision as sv
from collections import defaultdict, deque
import mysql.connector
from mysql.connector import Error
import xml.etree.ElementTree as ET
import pandas as pd
import time
from typing import Tuple, Dict

# Add DB configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'Traffic1',
    'password': 'pedestrian',
    'database': 'traffic_db'
}

def read_time_from_xml(xml_path: str, frame_count: int, fps: float = 25.0) -> datetime:
    """Read time from XML file and calculate current time based on frame count"""
    try:
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get creation date from XML
            creation_date = root.find('.//{*}CreationDate')
            if creation_date is not None and 'value' in creation_date.attrib:
                timestamp_str = creation_date.get('value').split('+')[0]
                print(f"\nFound XML timestamp: {timestamp_str}")
                
                # Parse base time from XML
                base_time = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
                print(f"Parsed base time: {base_time}")
                
                # Calculate current time based on frame count
                print(f"Frame count: {frame_count}")
                print(f"FPS: {fps}")
                seconds_offset = frame_count / fps
                print(f"Seconds offset: {seconds_offset}")
                
                xml_time = base_time + timedelta(seconds=seconds_offset)
                print(f"XML time: {xml_time}")
                return xml_time
            else:
                print("Warning: No CreationDate found in XML, using system time")
                return datetime.now()
        else:
            print(f"Warning: XML file not found at {xml_path}, using system time")
            return datetime.now()
            
    except Exception as e:
        print(f"Error reading XML time: {e}")
        print("Using system time as fallback")
        return datetime.now()

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

class PedestrianPipeline:
    def __init__(self, model_path: str, pixels_per_meter: float, processors: dict, conf_threshold: float = 0.7):
        """Initialize the main pipeline"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Store configuration
            self.processors = processors
            self.pixels_per_meter = pixels_per_meter
            self.conf_threshold = conf_threshold
            
            # Get number of lanes from user if PCA is enabled
            self.num_lanes = None
            if self.processors['PCA']:
                while True:
                    try:
                        self.num_lanes = int(input("Enter the number of lanes (minimum 1): "))
                        if self.num_lanes >= 1:
                            break
                        print("Please enter a number greater than or equal to 1")
                    except ValueError:
                        print("Please enter a valid number")
            
            # Initialize YOLO model
            self.model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")
            
            # Initialize database connection
            self.video_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            try:
                self.db_connection = mysql.connector.connect(**DB_CONFIG)
                self.db_cursor = self.db_connection.cursor()
                print("Successfully connected to MySQL database")
                self.setup_database_tables()
                self.setup_sample_data()
                print("Sample data setup complete")
            except Error as e:
                print(f"Error connecting to MySQL database: {e}")
                self.db_connection = None
                self.db_cursor = None
            
            # Initialize separate trackers for pedestrians and vehicles
            self.ped_tracker = Sort(
                max_age=60,
                min_hits=3,
                iou_threshold=0.3
            )
            
            self.vehicle_tracker = Sort(
                max_age=60,
                min_hits=3,
                iou_threshold=0.3
            )
            
            # Initialize ID counters
            self.next_ped_id = 1
            self.next_vehicle_id = 1
            
            # Add appearance history for re-identification
            self.appearance_history = {}
            self.ped_id_mapping = {}
            self.vehicle_id_mapping = {}
            self.vehicle_classes = {}
            self.temp_vehicle_classes = {}
            
            # Initialize coordinates history for vehicle tracking
            self.coordinates = defaultdict(lambda: deque(maxlen=60))  # 2 seconds history at 30fps
            
            # Initialize SOURCE and TARGET points (will be set during video processing)
            self.SOURCE = None
            self.TARGET = None
            self.view_transformer = None
            
            # Initialize ultralytics components
            self.ultra_model = YOLO("best(3).pt")
            self.byte_track = sv.ByteTrack(
                frame_rate=30,  # Default FPS, will be updated when video is loaded
                track_activation_threshold=conf_threshold
            )
            
            # Setup CSV paths
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs('Synched Pipeline/logs', exist_ok=True)
            
            # Create and store CSV paths with unique names
            self.ped_csv = f'Synched Pipeline/logs/pedestrian_analysis_{timestamp}.csv'
            self.vehicle_csv = f'Synched Pipeline/logs/vehicle_analysis_{timestamp}.csv'
            self.vcomp_csv = f'Synched Pipeline/logs/vehicle_counts_{timestamp}.csv'
            self.ultra_csv = f'Synched Pipeline/logs/ultralytics_analysis_{timestamp}.csv'
            
            print(f"CSV files will be saved to Synched Pipeline/logs directory with timestamp: {timestamp}")
            
            # Initialize only enabled processors
            if self.processors['PS']:
                print(f"Pedestrian analysis CSV: {self.ped_csv}")
                self.speed_detector = PedestrianSpeedDetector(pixels_per_meter, self.ped_csv)
                self.speed_detector.pipeline = self  # Add pipeline reference
                print("Speed Detector initialized with pipeline reference.")
            
            if self.processors['PCA']:
                self.crossing_analyzer = PedestrianCrossingAnalyzer(self.ped_csv, self.num_lanes)
                print("Crossing Analyzer initialization complete.")
                
            if self.processors['CARPSM']:
                print(f"Vehicle analysis CSV: {self.vehicle_csv}")
                self.car_detector = CarPSMDetector(model_path, self.vehicle_csv)
                self.car_detector.set_pipeline(self)  # Pass pipeline reference to CARPSM detector
                print("CarPSMDetector initialization complete.")
                
            if self.processors['VCOMP']:
                print(f"Vehicle counts CSV: {self.vcomp_csv}")
                self.vehicle_analyzer = VehicleAnalyzer(self.vcomp_csv)
                self.vehicle_analyzer.set_database_connection(self.db_connection, self.db_cursor)
                print("VehicleAnalyzer initialization complete.")
                
            if self.processors['Ultralytics']:
                print(f"Ultralytics analysis CSV: {self.ultra_csv}")
                self.ultralytics_processor = UltralyticsProcessor(self.ultra_csv, model_path, conf_threshold)
                print("UltralyticsProcessor initialization complete.")
            
            print("Pipeline initialization complete.")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def setup_database_tables(self):
        """Setup necessary database tables"""
        try:
            # Create tables if they don't exist
            tables = [
                """CREATE TABLE IF NOT EXISTS time_dimension (
                    time_key INTEGER PRIMARY KEY,
                    week VARCHAR(20),
                    day VARCHAR(20),
                    day_night VARCHAR(10),
                    date DATE,
                    hour INTEGER,
                    minute INTEGER
                )""",
                
                """CREATE TABLE IF NOT EXISTS scene_dimension (
                    scene_key INTEGER,
                    road_key INTEGER,
                    object_type VARCHAR(50),
                    event_type VARCHAR(50),
                    PRIMARY KEY (scene_key, road_key),
                    FOREIGN KEY (road_key) REFERENCES road_character_dimension(road_key)
                )""",
                
                """CREATE TABLE IF NOT EXISTS behavior_feature (
                    behavior_id INTEGER PRIMARY KEY,
                    behavior_feature VARCHAR(255)
                )""",
                
                """CREATE TABLE IF NOT EXISTS fact_table (
                    time_key INTEGER,
                    location_key INTEGER,
                    road_key INTEGER,
                    scene_key INTEGER,
                    scene_ratio FLOAT,
                    video_code VARCHAR(255),
                    PRIMARY KEY (time_key, location_key, road_key, scene_key),
                    FOREIGN KEY (scene_key, road_key) REFERENCES scene_dimension(scene_key, road_key),
                    FOREIGN KEY (time_key) REFERENCES time_dimension(time_key),
                    FOREIGN KEY (location_key) REFERENCES location_dimension(location_key)
                )""",
                
                """CREATE TABLE IF NOT EXISTS vehicle_traffic (
                    time_key INTEGER,
                    hour INTEGER,
                    minute_interval INTEGER,  # 0-5 for 10-minute intervals
                    pedestrian_count INTEGER DEFAULT 0,
                    car_count INTEGER DEFAULT 0,
                    bus_count INTEGER DEFAULT 0,
                    truck_count INTEGER DEFAULT 0,
                    PRIMARY KEY (time_key, minute_interval),
                    FOREIGN KEY (time_key) REFERENCES time_dimension(time_key)
                )"""
            ]
            
            for table in tables:
                try:
                    self.db_cursor.execute(table)
                    print(f"Successfully created/verified table")
                except Error as e:
                    print(f"Error creating table: {e}")
                    raise

            # Insert behavior features if they don't exist
            behaviors = [
                # Vehicle behaviors (IDs 1-6)
                (1, 'vehicle_acceleration'),
                (2, 'vehicle_speed_5m'),
                (3, 'vehicle_speed_10m'),
                (4, 'vehicle_speed_15m'),
                (5, 'vehicle_avg_speed'),
                (6, 'vehicle_class'),
                
                # Pedestrian behaviors (IDs 10-15)
                (10, 'pedestrian_parking'),
                (11, 'pedestrian_direction'),
                (12, 'pedestrian_speed'),
                (13, 'pedestrian_vehicle_type'),
                (14, 'pedestrian_psm'),
                (15, 'pedestrian_parking')
            ]
            
            for behavior_id, behavior_feature in behaviors:
                try:
                    self.db_cursor.execute(
                        """INSERT INTO behavior_feature (behavior_id, behavior_feature) 
                           VALUES (%s, %s) 
                           ON DUPLICATE KEY UPDATE behavior_feature = VALUES(behavior_feature)""",
                        (behavior_id, behavior_feature)
                    )
                except Error as e:
                    print(f"Error inserting behavior feature: {e}")
                    raise
            
            # Commit all changes
            self.db_connection.commit()
            print("Database tables setup complete")
            
        except Error as e:
            print(f"Error setting up database tables: {e}")
            self.db_connection.rollback()
            raise

    def insert_track_data(self, track_id: int, object_type: str, speeds: dict = None, class_id: int = None, 
                         vehicle_type: str = None, time_difference: float = None):
        """Insert track data into database"""
        if not self.db_connection or not self.db_cursor:
            print("Warning: Database connection not available. Cannot insert track data.")
            return
            
        try:
            # Ensure all values are proper Python types
            track_id = int(track_id)
            
            # Get current frame count from video capture
            if hasattr(self, 'cap'):
                frame_count = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                frame_count = 0
                
            # Get XML time based on frame count
            current_time = read_time_from_xml(frame_count=frame_count, fps=self.fps)
            print(f"Using XML time for frame {frame_count}: {current_time}")
            
            # Print debug information
            print(f"Inserting track data for {object_type} ID {track_id}")
            if speeds:
                print(f"Speeds to insert: {speeds}")
            if vehicle_type:
                print(f"Vehicle type to insert: {vehicle_type}")
            if time_difference:
                print(f"PSM/Time difference to insert: {time_difference}")
            
            # Insert time dimension with XML-based time
            self.db_cursor.execute(
                """INSERT IGNORE INTO time_dimension 
                   (time_key, week, day, day_night, date, hour, minute)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    track_id,
                    f"Week{current_time.strftime('%V')}",
                    current_time.strftime('%A'),
                    'Day' if 6 <= current_time.hour <= 18 else 'Night',
                    current_time.date(),
                    current_time.hour,
                    current_time.minute
                )
            )
            print(f"Time dimension inserted with XML time: {current_time}")
            
            # Insert scene dimension with road_key=3
            self.db_cursor.execute(
                """INSERT IGNORE INTO scene_dimension
                   (scene_key, road_key, object_type, event_type)
                   VALUES (%s, %s, %s, %s)""",
                (track_id, 3, str(object_type), 'movement')
            )
            print("Scene dimension inserted with road_key=3")
            
            # Insert fact table with location_key=3 and road_key=3
            self.db_cursor.execute(
                """INSERT IGNORE INTO fact_table
                   (time_key, location_key, road_key, scene_key, scene_ratio, video_code)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                   scene_ratio = VALUES(scene_ratio),
                   video_code = VALUES(video_code)""",
                (track_id, 3, 3, track_id, 1.0, f"video_{self.video_id}_frame{frame_count}_{track_id}")
            )
            print("Fact table inserted with location_key=3, road_key=3")
            
            # Handle speeds and direction separately
            if speeds:
                for behavior_id, value in speeds.items():
                    behavior_id = int(behavior_id)
                    value = float(value)
                    
                    # Direction (behavior_id 11) should not be subject to speed threshold
                    if behavior_id == 11:
                        print(f"Inserting direction value {value} for {object_type} {track_id}")
                        self.db_cursor.execute(
                            """INSERT INTO scene_behavior_feature
                               (scene_key, object_type, behavior_id, behavior_value)
                               VALUES (%s, %s, %s, %s)
                               ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)""",
                            (track_id, str(object_type), behavior_id, value)
                        )
                        print(f"Direction value {value} inserted for behavior_id {behavior_id}")
                    # For actual speeds, apply the threshold
                    elif value < 3.0:
                        print(f"Skipping speed {value:.2f} km/h for behavior_id {behavior_id} (below threshold)")
                        continue
                    else:
                        print(f"Inserting speed {value:.2f} km/h for behavior_id {behavior_id}")
                        self.db_cursor.execute(
                            """INSERT INTO scene_behavior_feature
                               (scene_key, object_type, behavior_id, behavior_value)
                               VALUES (%s, %s, %s, %s)
                               ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)""",
                            (track_id, str(object_type), behavior_id, value)
                        )
                        print(f"Speed {value:.2f} km/h inserted for behavior_id {behavior_id}")
            
            # Insert vehicle class if provided
            if class_id is not None:
                class_id = int(class_id)
                print(f"Inserting vehicle class {class_id} for track {track_id}")
                self.db_cursor.execute(
                    """INSERT INTO scene_behavior_feature
                       (scene_key, object_type, behavior_id, behavior_value)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)""",
                    (track_id, str(object_type), 6, float(class_id))  # behavior_id 6 for vehicle_class
                )
                print("Vehicle class inserted")
            
            # Insert vehicle type if provided (for pedestrians)
            if vehicle_type is not None and object_type == 'pedestrian':
                print(f"Inserting vehicle type {vehicle_type} for pedestrian {track_id}")
                vehicle_type_id = self.get_vehicle_type_id(vehicle_type)
                self.db_cursor.execute(
                    """INSERT INTO scene_behavior_feature
                       (scene_key, object_type, behavior_id, behavior_value)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)""",
                    (track_id, str(object_type), 13, float(vehicle_type_id))  # behavior_id 13 for pedestrian_vehicle_type
                )
                print(f"Vehicle type {vehicle_type} (ID: {vehicle_type_id}) inserted for pedestrian")
            
            # Insert PSM/time difference if provided (for pedestrians)
            if time_difference is not None and object_type == 'pedestrian':
                print(f"Inserting PSM/time difference {time_difference} seconds for pedestrian {track_id}")
                self.db_cursor.execute(
                    """INSERT INTO scene_behavior_feature
                       (scene_key, object_type, behavior_id, behavior_value)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)""",
                    (track_id, str(object_type), 14, float(time_difference))  # behavior_id 14 for PSM/time difference
                )
                print("PSM/time difference inserted")
            
            # Commit the transaction
            self.db_connection.commit()
            print(f"Successfully committed track data for {object_type} ID {track_id} at frame {frame_count}")
            
        except Exception as e:
            print(f"Error inserting track data: {e}")
            self.db_connection.rollback()
            # Print more detailed error information
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_cursor') and self.db_cursor:
            self.db_cursor.close()
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()

    def select_roi(self, frame):
        """Allow user to select region of interest for bird's eye view"""
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
                cv2.imshow("Select ROI", frame_copy)
        
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select ROI", mouse_callback)
        
        print("\nInstructions:")
        print("1. Click to select 4 points to define the road area")
        print("2. Press 'q' when done or 'r' to reset")
        
        while True:
            cv2.imshow("Select ROI", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(points) == 4:
                break
            elif key == ord('r'):
                points = []
                frame_copy = frame.copy()
                cv2.imshow("Select ROI", frame_copy)
        
        cv2.destroyAllWindows()
        return np.array(points)

    def setup_birds_eye_view(self, first_frame):
        """Setup bird's eye view transformation"""
        # Only setup if bird's eye view is enabled
        if not self.processors['BirdEye']:
            # Set default values
            self.TARGET_WIDTH = 800
            self.TARGET_HEIGHT = 600
            self.SOURCE = np.array([[0, 0], [800, 0], [800, 600], [0, 600]])
            self.TARGET = np.array([[0, 0], [800, 0], [800, 600], [0, 600]])
            self.view_transformer = ViewTransformer(source=self.SOURCE, target=self.TARGET)
            return self.TARGET_WIDTH, self.TARGET_HEIGHT

        # Select ROI points
        self.SOURCE = self.select_roi(first_frame)
        print(f"Selected ROI points: {self.SOURCE}")
        
        # Define target dimensions
        REAL_WIDTH = 17  # meters
        REAL_HEIGHT = 30  # meters
        self.TARGET_WIDTH = int(REAL_WIDTH * self.pixels_per_meter)
        self.TARGET_HEIGHT = int(REAL_HEIGHT * self.pixels_per_meter)
        
        # Define target points
        self.TARGET = np.array([
            [0, 0],
            [self.TARGET_WIDTH - 1, 0],
            [self.TARGET_WIDTH - 1, self.TARGET_HEIGHT - 1],
            [0, self.TARGET_HEIGHT - 1],
        ])
        
        # Initialize view transformer
        self.view_transformer = ViewTransformer(source=self.SOURCE, target=self.TARGET)
        
        return self.TARGET_WIDTH, self.TARGET_HEIGHT

    def process_video(self, video_path: str):
        """Process video file and coordinate external processing"""
        # Normalize path and check existence
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"\nProcessing video: {video_path}")
        print(f"Video directory: {os.path.dirname(video_path)}")
        
        start_time = time.time()
        last_update_time = start_time
        update_interval = 5  # seconds
        
        try:
            # Initialize video capture and store as instance variable
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties with validation
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if width <= 0 or height <= 0:
                raise ValueError("Invalid video dimensions")
            if self.fps <= 0:
                self.fps = 25  # Default FPS
                print("Warning: Could not determine FPS, using default value of 25")
            
            print(f"\nVideo properties:")
            print(f"- Dimensions: {width}x{height}")
            print(f"- FPS: {self.fps}")
            print(f"- Total frames: {total_frames}")
            print(f"- Duration: {total_frames/self.fps:.2f} seconds")
            
            # Update byte track frame rate
            self.byte_track = sv.ByteTrack(
                frame_rate=self.fps,
                track_activation_threshold=self.conf_threshold
            )
            
            # Initialize supervision annotators
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=(width, height))
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=(width, height))
            box_annotator = sv.BoxAnnotator(thickness=thickness)
            label_annotator = sv.LabelAnnotator(
                text_scale=text_scale,
                text_thickness=thickness,
                text_position=sv.Position.BOTTOM_CENTER,
            )
            trace_annotator = sv.TraceAnnotator(
                thickness=thickness,
                trace_length=self.fps * 2,
                position=sv.Position.BOTTOM_CENTER,
            )
            
            # Get first frame for setup
            ret, first_frame = self.cap.read()
            if not ret or first_frame is None:
                raise ValueError("Could not read first frame")
            
            print("\nSetting up processing components...")
            
            # Setup bird's eye view
            target_width, target_height = self.setup_birds_eye_view(first_frame)
            print("Bird's eye view setup complete")
            
            # Select lanes and target y-coordinate for car detection if CARPSM is enabled
            if self.processors['PCA']:
                print("\nSelect points for pedestrian crossing analysis:")
                self.crossing_analyzer.select_points(first_frame)
                if self.processors['CARPSM']:
                    print("\nSelect target y-coordinate for car detection:")
                    self.car_detector.lanes = self.crossing_analyzer.lanes
                    self.car_detector.select_target_y(first_frame)
            
            # Select reference points for Ultralytics processor if enabled
            if self.processors['Ultralytics']:
                print("\nSelect reference points for Ultralytics processor:")
                self.ultralytics_processor.select_reference_points(first_frame)
            
            # Select vehicle counting region if VCOMP is enabled
            if self.processors['VCOMP']:
                print("\nSelect region for vehicle counting:")
                self.vehicle_analyzer.select_count_region(first_frame)
            
            # Reset video capture to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create output directory
            os.makedirs('output', exist_ok=True)
            print("\nInitializing video writers...")
            
            # Initialize video writers for enabled processors
            writers = {}
            if any(self.processors.values()):  # Only create writers if any processor is enabled
                writers['ped'] = cv2.VideoWriter(
                    os.path.join('output', f"processed_{os.path.basename(video_path)}"),
                    cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height)
                )
                
                if self.processors['CARPSM']:
                    writers['car'] = cv2.VideoWriter(
                        os.path.join('output', f"carpsm_{os.path.basename(video_path)}"),
                        cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height)
                    )
                
                if self.processors['VCOMP']:
                    writers['vcomp'] = cv2.VideoWriter(
                        os.path.join('output', f"vcomp_{os.path.basename(video_path)}"),
                        cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height)
                    )
                
                if self.processors['BirdEye']:
                    writers['birds_eye'] = cv2.VideoWriter(
                        os.path.join('output', f"birds_eye_{os.path.basename(video_path)}"),
                        cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (target_width, target_height)
                    )
                
                if self.processors['Ultralytics']:
                    writers['ultra'] = cv2.VideoWriter(
                        os.path.join('output', f"ultralytics_{os.path.basename(video_path)}"),
                        cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height)
                    )
            
            print("Video writers initialized")
            print("\nStarting video processing...")
            
            frame_count = 0
            while True:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    current_time = read_time_from_xml(frame_count=frame_count, fps=self.fps).strftime('%H:%M:%S')
                    frame_time = float(frame_count) / float(self.fps)
                    
                    try:
                        # Get detections from YOLO model with NMS
                        results = self.model(frame, conf=self.conf_threshold, iou=0.5)[0]
                        
                        # Process detections and get tracked objects
                        ped_tracks, vehicle_tracks = self.process_detections(results, frame)
                        
                        # Process frames through enabled components
                        processed_frames = {}
                        
                        try:
                            # Main pedestrian frame
                            ped_frame = frame.copy()
                            if self.processors['PCA']:
                                ped_frame = self.crossing_analyzer.process_frame(
                                    ped_frame, ped_tracks, frame_time, vehicle_tracks, current_time)
                            if self.processors['PS']:
                                ped_frame = self.speed_detector.process_frame(
                                    ped_frame, ped_tracks, frame_time)
                            processed_frames['ped'] = ped_frame
                        except Exception as e:
                            print(f"Error processing pedestrian frame: {e}")
                            processed_frames['ped'] = frame.copy()
                        
                        try:
                            # Car PSM frame
                            if self.processors['CARPSM']:
                                car_frame = frame.copy()
                                processed_frames['car'] = self.car_detector.process_frame(
                                    car_frame, vehicle_tracks, current_time)
                        except Exception as e:
                            print(f"Error processing car frame: {e}")
                            if self.processors['CARPSM']:
                                processed_frames['car'] = frame.copy()
                        
                        try:
                            # Vehicle composition frame
                            if self.processors['VCOMP']:
                                vcomp_frame = frame.copy()
                                processed_frames['vcomp'] = self.vehicle_analyzer.process_frame(
                                    vcomp_frame, vehicle_tracks, self.vehicle_classes, frame_time)
                        except Exception as e:
                            print(f"Error processing vehicle composition frame: {e}")
                            if self.processors['VCOMP']:
                                processed_frames['vcomp'] = frame.copy()
                        
                        try:
                            # Bird's eye view frame
                            if self.processors['BirdEye']:
                                birds_eye_frame, visible_ids = self.process_birds_eye_view(frame, vehicle_tracks)
                                processed_frames['birds_eye'] = birds_eye_frame
                            else:
                                visible_ids = set()
                        except Exception as e:
                            print(f"Error processing bird's eye view frame: {e}")
                            visible_ids = set()
                        
                        try:
                            # Ultralytics frame
                            if self.processors['Ultralytics']:
                                ultra_frame = frame.copy()
                                processed_frames['ultra'] = self.ultralytics_processor.process_frame(
                                    ultra_frame, vehicle_tracks, visible_ids)
                        except Exception as e:
                            print(f"Error processing ultralytics frame: {e}")
                            if self.processors['Ultralytics']:
                                processed_frames['ultra'] = frame.copy()
                        
                        # Add timestamp and frame info to all frames
                        for key, processed_frame in processed_frames.items():
                            if key != 'birds_eye':  # Don't add to bird's eye view
                                cv2.putText(processed_frame, f"Time: {current_time}", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                cv2.putText(processed_frame, f"Frame: {frame_count} | FPS: {self.fps:.1f}", 
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Write frames to video files
                        for key, processed_frame in processed_frames.items():
                            if key in writers:
                                writers[key].write(processed_frame)
                        
                        # Display frames
                        for key, processed_frame in processed_frames.items():
                            window_name = {
                                'ped': 'Traffic Analysis',
                                'car': 'Car PSM',
                                'vcomp': 'Vehicle Composition',
                                'birds_eye': "Bird's Eye View",
                                'ultra': 'Ultralytics Processor'
                            }.get(key, key)
                            cv2.imshow(window_name, processed_frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nProcessing interrupted by user")
                            break
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_count}:")
                        print(f"Error details: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Write original frame if processing fails
                        for writer in writers.values():
                            writer.write(frame)
                    
                    # Update progress
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval:
                        progress = (frame_count / total_frames) * 100
                        elapsed_time = current_time - start_time
                        fps = frame_count / elapsed_time
                        eta = (total_frames - frame_count) / fps
                        
                        print(f"\nProgress update:")
                        print(f"- Processed frames: {frame_count}/{total_frames} ({progress:.1f}%)")
                        print(f"- Average FPS: {fps:.1f}")
                        print(f"- Elapsed time: {elapsed_time:.1f}s")
                        print(f"- Estimated time remaining: {eta:.1f}s")
                        
                        last_update_time = current_time
                
                except Exception as e:
                    print(f"Critical error processing frame {frame_count}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # Final statistics
            total_time = time.time() - start_time
            average_fps = frame_count / total_time
            print(f"\nProcessing complete!")
            print(f"- Total frames processed: {frame_count}")
            print(f"- Total time: {total_time:.1f}s")
            print(f"- Average FPS: {average_fps:.1f}")
            
            # Save final results
            print("\nSaving final results...")
            if self.processors['VCOMP']:
                self.vehicle_analyzer.save_results_to_csv()
                self.vehicle_analyzer.print_statistics()
            if self.processors['Ultralytics']:
                self.ultralytics_processor.save_results_to_csv()
                self.ultralytics_processor.print_statistics()
            
            # Print output paths
            print("\nOutput files:")
            for key, writer in writers.items():
                output_path = os.path.join('output', f"{key}_{os.path.basename(video_path)}")
                print(f"- {key}: {os.path.abspath(output_path)}")
            
            print("\nCSV files:")
            if self.processors['PS']:
                print(f"- Pedestrian analysis: {os.path.abspath(self.ped_csv)}")
            if self.processors['CARPSM']:
                print(f"- Vehicle analysis: {os.path.abspath(self.vehicle_csv)}")
            if self.processors['VCOMP']:
                print(f"- Vehicle counts: {os.path.abspath(self.vcomp_csv)}")
            if self.processors['Ultralytics']:
                print(f"- Ultralytics analysis: {os.path.abspath(self.ultra_csv)}")
            
            # Process PSM results
            print("\nProcessing PSM results...")
            psm_results_path = os.path.join('results', 'pedestrian_vehicle_analysis.csv')
            if os.path.exists(psm_results_path):
                self.process_psm_results(psm_results_path)
            else:
                print(f"PSM results file not found at: {psm_results_path}")
            
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup resources
            try:
                if 'cap' in locals():
                    self.cap.release()
                for writer in writers.values():
                    writer.release()
                cv2.destroyAllWindows()
                
                # Close database connection
                if hasattr(self, 'db_connection') and self.db_connection:
                    if hasattr(self, 'db_cursor') and self.db_cursor:
                        self.db_cursor.close()
                    self.db_connection.close()
                
                print("\nAll resources cleaned up successfully")
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
                import traceback
                traceback.print_exc()

    def process_birds_eye_view(self, frame, vehicle_tracks):
        """Process frame for bird's eye view"""
        # Create white background for bird's eye view
        birds_eye_frame = np.ones((self.TARGET_HEIGHT, self.TARGET_WIDTH, 3), dtype=np.uint8) * 255
        
        # Initialize set to track visible IDs
        visible_ids = set()
        
        # Draw grid with better visibility
        # Major grid lines (every 5 meters)
        for x in range(0, self.TARGET_WIDTH, 5 * self.pixels_per_meter):
            cv2.line(birds_eye_frame, (x, 0), (x, self.TARGET_HEIGHT), (200, 200, 200), 2)
            # Add distance markers in meters with much larger text
            meters = x / self.pixels_per_meter
            cv2.putText(birds_eye_frame, f"{meters:.1f}m", (x+2, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)

        for y in range(0, self.TARGET_HEIGHT, 5 * self.pixels_per_meter):
            cv2.line(birds_eye_frame, (0, y), (self.TARGET_WIDTH, y), (200, 200, 200), 2)
            # Add distance markers in meters with much larger text
            meters = y / self.pixels_per_meter
            cv2.putText(birds_eye_frame, f"{meters:.1f}m", (5, y+40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)

        # Minor grid lines (every 1 meter)
        for x in range(0, self.TARGET_WIDTH, self.pixels_per_meter):
            cv2.line(birds_eye_frame, (x, 0), (x, self.TARGET_HEIGHT), (230, 230, 230), 1)
        for y in range(0, self.TARGET_HEIGHT, self.pixels_per_meter):
            cv2.line(birds_eye_frame, (0, y), (self.TARGET_WIDTH, y), (230, 230, 230), 1)

        # Draw border
        cv2.rectangle(birds_eye_frame, (0, 0), (self.TARGET_WIDTH-1, self.TARGET_HEIGHT-1), (0, 0, 0), 2)
        
        # Initialize reference lines if not already done
        if not hasattr(self, 'reference_lines'):
            self.reference_lines = {}
            if hasattr(self.ultralytics_processor, 'reference_line'):
                main_ref_line = np.array(self.ultralytics_processor.reference_line)
                transformed_main = self.view_transformer.transform_points(main_ref_line)
                if len(transformed_main) == 2:
                    # Store the main reference line
                    self.reference_lines['main'] = transformed_main
                    
                    # Calculate direction vector of the line
                    direction = transformed_main[1] - transformed_main[0]
                    unit_direction = direction / np.linalg.norm(direction)
                    
                    # Create additional reference lines at 5m intervals
                    for distance in [5, 10, 15]:
                        # Calculate offset vector (perpendicular to line direction)
                        offset = np.array([unit_direction[1], -unit_direction[0]]) * (distance * self.pixels_per_meter)
                        
                        # Create new line by offsetting both points
                        new_line = np.array([
                            transformed_main[0] + offset,
                            transformed_main[1] + offset
                        ])
                        
                        self.reference_lines[f'{distance}m'] = new_line

        # Draw all reference lines
        colors = {
            'main': (0, 255, 255),  # Yellow for main line
            '15m': (0, 255, 0),     # Green
            '10m': (255, 128, 0),   # Orange
            '5m': (255, 0, 0)       # Red
        }
        
        # Initialize speed tracking if not exists
        if not hasattr(self, 'vehicle_speeds'):
            self.vehicle_speeds = {}  # Format: {track_id: {line_key: speed}}
        
        # Draw reference lines and labels
        for line_key, line in self.reference_lines.items():
            p1, p2 = line.astype(int)
            color = colors.get(line_key, (0, 255, 255))
            cv2.line(birds_eye_frame, tuple(p1), tuple(p2), color, 2)
            
            # Add label for each line
            label = "Speed Measurement Line" if line_key == 'main' else f"Reference Line {line_key}"
            cv2.putText(birds_eye_frame, label, 
                       (p1[0], p1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Process each vehicle track
        for track in vehicle_tracks:
            x1, y1, x2, y2, track_id = track
            # Get bottom center point
            bottom_center = np.array([[(x1 + x2) / 2, y2]])
            
            # Transform point to bird's eye view
            transformed_point = self.view_transformer.transform_points(bottom_center)[0]
            
            # Skip if point is outside the target area
            if not (0 <= transformed_point[0] < self.TARGET_WIDTH and 
                   0 <= transformed_point[1] < self.TARGET_HEIGHT):
                continue
            
            # Add ID to visible set since it's in the bird's eye view
            visible_ids.add(track_id)
            
            # Record y-coordinate
            self.coordinates[track_id].append(transformed_point)
            
            # Draw vehicle position with better visibility
            cv2.circle(birds_eye_frame, (int(transformed_point[0]), int(transformed_point[1])), 
                      8, (0, 0, 255), -1)  # Red dot for current position
            
            # Draw ID with much larger text
            cv2.putText(birds_eye_frame, f"#{int(track_id)}", 
                       (int(transformed_point[0]) + 5, int(transformed_point[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Initialize speeds dictionary for this vehicle if not exists
            if track_id not in self.vehicle_speeds:
                self.vehicle_speeds[track_id] = {}
            
            # Draw trails with fading effect
            points_list = list(self.coordinates[track_id])
            if len(points_list) >= 2:
                for i in range(1, len(points_list)):
                    # Draw trail with fading effect
                    cv2.line(birds_eye_frame, 
                            (int(points_list[i-1][0]), int(points_list[i-1][1])), 
                            (int(points_list[i][0]), int(points_list[i][1])),
                            (0, 0, int(255 * (i / len(points_list)))), 3)
                    
                    # Check intersection with all reference lines
                    if len(points_list) >= self.fps / 2:
                        def ccw(A, B, C):
                            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
                        
                        for line_key, ref_line in self.reference_lines.items():
                            # Skip if we already have speed for this line
                            if line_key in self.vehicle_speeds[track_id]:
                                continue
                                
                            p1, p2 = ref_line
                            # Check if the line segments intersect
                            if (ccw(points_list[i-1], p1, p2) != ccw(points_list[i], p1, p2) and
                                ccw(points_list[i-1], points_list[i], p1) != ccw(points_list[i-1], points_list[i], p2)):
                                
                                # Calculate speed at crossing point
                                start_y = float(points_list[0][1] / self.pixels_per_meter)  # Convert to meters
                                end_y = float(points_list[-1][1] / self.pixels_per_meter)   # Convert to meters
                                distance = float(abs(end_y - start_y))  # Distance in meters
                                time = float(len(points_list) / self.fps)     # Time in seconds
                                speed = float(distance / time * 3.6)          # Convert to km/h
                                
                                # Store the speed for this vehicle at this line
                                self.vehicle_speeds[track_id][line_key] = speed
                                
                                # Log speed to SQL database with appropriate behavior_id
                                behavior_ids = {
                                    'main': 1,  # Reference_Speed
                                    '5m': 2,    # Speed_5m
                                    '10m': 3,   # Speed_10m
                                    '15m': 4    # Speed_15m
                                }
                                
                                if line_key in behavior_ids:
                                    # Print debug information
                                    print(f"Inserting speed for vehicle {track_id} at {line_key}: {speed:.2f} km/h")
                                    
                                    # Convert track_id to int if it's numpy type
                                    track_id_int = int(float(track_id))
                                    
                                    # Create speeds dictionary with Python native float
                                    speeds_dict = {behavior_ids[line_key]: float(speed)}
                                    
                                    # Get class_id and ensure it's a Python native type
                                    class_id = self.vehicle_classes.get(track_id)
                                    if class_id is not None:
                                        class_id = int(class_id)
                                    
                                    # Insert the speed into the database
                                    try:
                                        self.insert_track_data(
                                            track_id=track_id_int,
                                            object_type='vehicle',
                                            speeds=speeds_dict,
                                            class_id=class_id
                                        )
                                        print(f"Successfully inserted speed {speed:.2f} km/h for vehicle {track_id_int}")
                                    except Exception as e:
                                        print(f"Error inserting speed data: {e}")
                                        print(f"Debug info - track_id type: {type(track_id_int)}, "
                                              f"speed type: {type(list(speeds_dict.values())[0])}, "
                                              f"class_id type: {type(class_id) if class_id is not None else None}")
            
            # Display all calculated speeds for this vehicle
            if track_id in self.vehicle_speeds:
                y_offset = 40
                for line_key, speed in self.vehicle_speeds[track_id].items():
                    color = colors.get(line_key, (0, 0, 255))
                    
                    # Skip display if speed is under 3 km/h
                    if speed < 3.0:
                        continue
                        
                    cv2.putText(birds_eye_frame, f"{line_key}: {speed:.1f} km/h",
                               (int(transformed_point[0]) + 5, int(transformed_point[1]) + y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    y_offset += 40
                
                # Check for acceleration/deceleration
                if 'main' in self.vehicle_speeds[track_id] and '5m' in self.vehicle_speeds[track_id]:
                    main_speed = self.vehicle_speeds[track_id]['main']
                    ref_speed = self.vehicle_speeds[track_id]['5m']
                    
                    # Only show label if both speeds are above 3 km/h
                    if main_speed >= 3.0 and ref_speed >= 3.0:
                        if ref_speed > main_speed:
                            label = "DECELERATION"
                            label_color = (0, 0, 255)  # Red
                            accel_value = 0  # 0 for deceleration
                        else:
                            label = "ACCELERATION"
                            label_color = (0, 255, 0)  # Green
                            accel_value = 1  # 1 for acceleration
                            
                        cv2.putText(birds_eye_frame, label,
                                   (int(transformed_point[0]) + 5, int(transformed_point[1]) + y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, label_color, 3)
                        
                        # Store acceleration/deceleration in the database
                        try:
                            # Convert track_id to int if it's numpy type
                            track_id_int = int(float(track_id))
                            
                            # Insert acceleration/deceleration data
                            self.insert_track_data(
                                track_id=track_id_int,
                                object_type='vehicle',
                                speeds={1: float(accel_value)},  # behavior_id 1 for vehicle_acceleration
                                class_id=self.vehicle_classes.get(track_id)
                            )
                            print(f"Inserted acceleration data for vehicle {track_id_int}: {label}")
                        except Exception as e:
                            print(f"Error inserting acceleration data: {e}")
        
        # Add title and scale information with much larger text
        cv2.putText(birds_eye_frame, "Bird's Eye View", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
        cv2.putText(birds_eye_frame, f"Scale: {self.pixels_per_meter}px = 1m",
                   (10, self.TARGET_HEIGHT - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        # Resize for better display
        display_height = 800
        display_width = int(self.TARGET_WIDTH * (display_height / self.TARGET_HEIGHT))
        birds_eye_frame = cv2.resize(birds_eye_frame, (display_width, display_height), 
                                   interpolation=cv2.INTER_AREA)
        
        return birds_eye_frame, visible_ids

    def calculate_bbox_features(self, frame, bbox):
        """Calculate simple features for a bounding box"""
        x1, y1, x2, y2 = map(int, bbox[:4])
        if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
            return None
        try:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            # Calculate average color and shape features
            avg_color = np.mean(roi, axis=(0, 1))
            aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
            area = (x2 - x1) * (y2 - y1)
            return np.array([*avg_color, aspect_ratio, area])
        except Exception:
            return None

    def match_features(self, current_features, history_features, threshold=0.85):
        """Match current features with historical features"""
        if current_features is None or history_features is None:
            return False
        try:
            # Normalize features
            current_norm = np.linalg.norm(current_features)
            history_norm = np.linalg.norm(history_features)
            if current_norm == 0 or history_norm == 0:
                return False
            current_features = current_features / current_norm
            history_features = history_features / history_norm
            # Calculate similarity
            similarity = np.dot(current_features, history_features)
            return similarity > threshold
        except Exception:
            return False

    def process_detections(self, results, frame):
        """Process YOLO detections and return tracked objects"""
        # Separate pedestrian and vehicle detections
        ped_detections = []
        vehicle_detections = []
        
        for det in results.boxes.data.tolist():
            # Ensure we have at least 6 elements (x1, y1, x2, y2, conf, cls)
            if len(det) < 6:
                continue
                
            x1, y1, x2, y2, conf, cls = det
            if conf > self.conf_threshold:
                if int(cls) == 3:  # Pedestrian class
                    ped_detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
                elif int(cls) in [0, 1, 2, 4, 5, 6]:  # Valid vehicle classes only
                    vehicle_detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
                    # Store the original class for this detection
                    bbox_key = (float(x1), float(y1), float(x2), float(y2))
                    self.temp_vehicle_classes[bbox_key] = int(cls)
        
        # Update tracker for pedestrians
        if len(ped_detections) > 0:
            ped_tracks = self.ped_tracker.update(np.array(ped_detections))
            
            # Process each track for consistent IDs
            for track in ped_tracks:
                if len(track) < 5:  # Ensure track has enough elements
                    continue
                    
                tracker_id = int(track[4])
                
                # Get or assign global ID
                if tracker_id not in self.ped_id_mapping:
                    self.ped_id_mapping[tracker_id] = self.next_ped_id
                    self.next_ped_id += 1
                
                # Update track with global ID
                track[4] = self.ped_id_mapping[tracker_id]
                
                # Update appearance history
                current_features = self.calculate_bbox_features(frame, track)
                if current_features is not None:
                    if track[4] not in self.appearance_history:
                        self.appearance_history[track[4]] = []
                    self.appearance_history[track[4]].append((track[:4], current_features))
                    # Keep only last 5 appearances
                    self.appearance_history[track[4]] = self.appearance_history[track[4]][-5:]
        else:
            ped_tracks = np.empty((0, 5))
            
        # Update tracker for vehicles
        if len(vehicle_detections) > 0:
            vehicle_tracks = self.vehicle_tracker.update(np.array(vehicle_detections))
            
            # Process each vehicle track
            for track in vehicle_tracks:
                if len(track) < 5:  # Ensure track has enough elements
                    continue
                    
                tracker_id = int(track[4])
                bbox = track[:4]
                
                # Get or assign global ID
                if tracker_id not in self.vehicle_id_mapping:
                    self.vehicle_id_mapping[tracker_id] = self.next_vehicle_id
                    self.next_vehicle_id += 1
                
                # Update track with global ID
                global_id = self.vehicle_id_mapping[tracker_id]
                track[4] = global_id
                
                # Find the closest matching detection to get its class
                closest_bbox = None
                min_dist = float('inf')
                track_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                
                for det_bbox in self.temp_vehicle_classes:
                    det_center = ((det_bbox[0] + det_bbox[2])/2, (det_bbox[1] + det_bbox[3])/2)
                    dist = ((track_center[0] - det_center[0])**2 + 
                           (track_center[1] - det_center[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_bbox = det_bbox
                
                # Update vehicle class if we found a close match
                if closest_bbox and min_dist < 50:  # 50 pixel threshold
                    self.vehicle_classes[global_id] = self.temp_vehicle_classes[closest_bbox]
        else:
            vehicle_tracks = np.empty((0, 5))
            
        # Clear temporary vehicle classes for next frame
        self.temp_vehicle_classes = {}
            
        return ped_tracks, vehicle_tracks

    def process_psm_results(self, psm_results_file):
        """Process PSM results and update database with time difference and vehicle type based on pedestrian IDs"""
        try:
            # Read PSM results
            psm_df = pd.read_csv(psm_results_file)
            
            # Process each row
            for _, row in psm_df.iterrows():
                pedestrian_id = row['Pedestrian_ID']
                time_diff = row['Time_Difference']
                vehicle_type = row['Vehicle_Type']
                direction = row['Direction']  # Get the direction from CSV
                
                # Convert time_diff to float, handling '<1' case
                time_diff_value = 0.5 if time_diff == '<1' else float(time_diff)
                
                # Convert direction to binary (0 or 1)
                # Assuming 'right_to_left' or similar positive direction is 1, and opposite is 0
                direction_value = 1.0 if direction.lower() in ['right_to_left', 'forward', 'positive'] else 0.0
                
                # Insert pedestrian data with time difference, vehicle type, and direction
                self.insert_track_data(
                    track_id=pedestrian_id,
                    object_type='pedestrian',
                    time_difference=time_diff_value,
                    vehicle_type=vehicle_type,  # Associate vehicle type with pedestrian ID
                    speeds={11: direction_value}  # Add direction as behavior_id 11
                )
                
                print(f"Processed PSM data for pedestrian {pedestrian_id}:")
                print(f"- Vehicle type: {vehicle_type}")
                print(f"- Direction: {direction} (value: {direction_value})")
                print(f"- Time difference: {time_diff_value}")
            
            print("Successfully processed PSM results and updated database")
            
        except Exception as e:
            print(f"Error processing PSM results: {e}")
            import traceback
            traceback.print_exc()

    def get_vehicle_type_id(self, vehicle_type):
        """Convert vehicle type string to numeric ID"""
        vehicle_type_map = {
            'Biker': 1,
            'Motorbike': 2,
            'Car': 3,
            'Taxi': 4,
            'Bus': 5,
            'Truck': 6,
            'Unknown': 0
        }
        return vehicle_type_map.get(vehicle_type, 0)

    def process_track(self, track_id: int, position: Tuple[float, float], frame_time: float, current_time: str = None) -> Dict:
        """Process a track and return crossing information"""
        # Get persistent ID for this track
        persistent_id = self.get_persistent_id(track_id)
        
        # Initialize crossing times for new track
        if persistent_id not in self.crossing_times:
            self.crossing_times[persistent_id] = {}
        
        # Check line crossings
        line_crossed = self.check_line_crossing(position, frame_time)
        if line_crossed is not None:
            # Use XML time if provided, otherwise use system time
            crossing_time = current_time if current_time else datetime.now().strftime('%H:%M:%S')
            
            # Record the time for this line crossing if not already recorded
            if line_crossed not in self.crossing_times[persistent_id]:
                self.crossing_times[persistent_id][line_crossed] = crossing_time
                
                # Calculate lane and line numbers based on the order of input
                actual_lane = (line_crossed // 3) + 1  # First lane is 1 (the first lane entered)
                line_in_lane = (line_crossed % 3) + 1  # Line position within lane (1-3)
                
                print(f"\nLine crossing detected:")
                print(f"Pedestrian {persistent_id} crossed line {line_crossed}")
                print(f"Lane {actual_lane}, Line {line_in_lane}")
                print(f"Time: {crossing_time}")
                
                # Save crossing to CSV immediately
                self.update_crossing_csv(track_id, line_crossed, crossing_time)
                
                # Determine direction based on lane crossing
                if actual_lane == 1 and line_in_lane == 1:
                    # Starting from first lane, first line = forward direction
                    direction_value = 1.0
                    print(f"Setting forward direction for pedestrian {persistent_id}")
                    try:
                        self.insert_track_data(
                            track_id=persistent_id,
                            object_type='pedestrian',
                            speeds={11: direction_value}  # behavior_id 11 for pedestrian_direction
                        )
                        print(f"Successfully inserted forward direction ({direction_value}) for pedestrian {persistent_id}")
                    except Exception as e:
                        print(f"Error inserting direction data: {e}")
                elif actual_lane == self.num_lanes and line_in_lane == 1:
                    # Starting from last lane, first line = reverse direction
                    direction_value = 0.0
                    print(f"Setting reverse direction for pedestrian {persistent_id}")
                    try:
                        self.insert_track_data(
                            track_id=persistent_id,
                            object_type='pedestrian',
                            speeds={11: direction_value}  # behavior_id 11 for pedestrian_direction
                        )
                        print(f"Successfully inserted reverse direction ({direction_value}) for pedestrian {persistent_id}")
                    except Exception as e:
                        print(f"Error inserting direction data: {e}")
        
        # Check and update parking violation status
        if persistent_id in self.parking_violations:
            try:
                print(f"Inserting parking violation for pedestrian {persistent_id}")
                self.db_cursor.execute("""
                    INSERT INTO scene_behavior_feature
                    (scene_key, object_type, behavior_id, behavior_value)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)
                """, (persistent_id, 'pedestrian', 15, 1.0))  # behavior_id 15 for parking violation
                self.db_connection.commit()
                print(f"Successfully inserted parking violation for pedestrian {persistent_id}")
            except Exception as e:
                print(f"Error inserting parking violation data: {e}")
                self.db_connection.rollback()
        else:
            try:
                print(f"Setting no parking violation for pedestrian {persistent_id}")
                self.db_cursor.execute("""
                    INSERT INTO scene_behavior_feature
                    (scene_key, object_type, behavior_id, behavior_value)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)
                """, (persistent_id, 'pedestrian', 15, 0.0))  # behavior_id 15 for no parking violation
                self.db_connection.commit()
                print(f"Successfully set no parking violation for pedestrian {persistent_id}")
            except Exception as e:
                print(f"Error setting no parking violation data: {e}")
                self.db_connection.rollback()
        
        return {
            'crossing_times': self.crossing_times.get(persistent_id, {})
        }

    def insert_vehicle_traffic(self, frame_count, counts):
        """Insert vehicle traffic data with proper time handling"""
        try:
            # Get XML time based on frame count
            current_time = read_time_from_xml(frame_count=frame_count, fps=self.fps)
            hour = current_time.hour
            minute = current_time.minute
            
            # Calculate 10-minute interval (0-5)
            # For example:
            # 15:37 -> minute=37 -> interval=3 (30-39 minutes)
            # 15:52 -> minute=52 -> interval=5 (50-59 minutes)
            minute_interval = min(minute // 10, 5)  # Ensure interval is 0-5
            interval_start = minute_interval * 10
            interval_end = min(interval_start + 9, 59)
            
            print(f"\nProcessing vehicle traffic data:")
            print(f"Current XML time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Hour: {hour:02d}:00")
            print(f"Current minute: {minute}")
            print(f"Calculated interval: {minute_interval} ({interval_start:02d}-{interval_end:02d} minutes)")
            
            # Generate time_key using date and hour
            time_key = int(f"{current_time.year}{current_time.month:02d}{current_time.day:02d}{hour:02d}")
            print(f"Generated time_key: {time_key}")
            
            # First ensure time dimension entry exists for this hour
            try:
                self.db_cursor.execute(
                    """INSERT IGNORE INTO time_dimension 
                       (time_key, week, day, day_night, date, hour, minute)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        time_key,
                        f"Week{current_time.strftime('%V')}",
                        current_time.strftime('%A'),
                        'Day' if 6 <= hour <= 18 else 'Night',
                        current_time.date(),
                        hour,
                        0  # Use 0 for the hour entry
                    )
                )
                print(f"Time dimension entry ensured for hour {hour:02d}:00")
                
                # Debug: Check if time dimension entry exists
                self.db_cursor.execute(
                    "SELECT * FROM time_dimension WHERE time_key = %s",
                    (time_key,)
                )
                time_entry = self.db_cursor.fetchone()
                if time_entry:
                    print(f"Confirmed time dimension entry: {time_entry}")
                
            except Exception as e:
                print(f"Error ensuring time dimension entry: {e}")
                raise
            
            # Now insert vehicle traffic data for this 10-minute interval
            try:
                print(f"\nInserting counts for interval {minute_interval}:")
                print(f"Pedestrians: {counts.get('pedestrian', 0)}")
                print(f"Cars: {counts.get('car', 0)}")
                print(f"Buses: {counts.get('bus', 0)}")
                print(f"Trucks: {counts.get('truck', 0)}")
                
                self.db_cursor.execute(
                    """INSERT INTO vehicle_traffic 
                       (time_key, hour, minute_interval, pedestrian_count, car_count, bus_count, truck_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                       pedestrian_count = pedestrian_count + VALUES(pedestrian_count),
                       car_count = car_count + VALUES(car_count),
                       bus_count = bus_count + VALUES(bus_count),
                       truck_count = truck_count + VALUES(truck_count)""",
                    (
                        time_key,
                        hour,
                        minute_interval,
                        counts.get('pedestrian', 0),
                        counts.get('car', 0),
                        counts.get('bus', 0),
                        counts.get('truck', 0)
                    )
                )
                
                # Debug: Check current intervals for this hour
                self.db_cursor.execute(
                    """SELECT minute_interval, pedestrian_count, car_count, bus_count, truck_count 
                       FROM vehicle_traffic 
                       WHERE time_key = %s AND hour = %s
                       ORDER BY minute_interval""",
                    (time_key, hour)
                )
                intervals = self.db_cursor.fetchall()
                print(f"\nCurrent intervals for hour {hour:02d}:00:")
                for interval in intervals:
                    interval_start = interval[0] * 10
                    interval_end = min(interval_start + 9, 59)
                    print(f"Interval {interval[0]} ({interval_start:02d}-{interval_end:02d} min): "
                          f"Ped={interval[1]}, Car={interval[2]}, Bus={interval[3]}, Truck={interval[4]}")
                
                self.db_connection.commit()
                print(f"\nSuccessfully updated vehicle traffic data")
                
            except Exception as e:
                print(f"Error inserting vehicle traffic data: {e}")
                self.db_connection.rollback()
                raise
            
        except Exception as e:
            print(f"Error in vehicle traffic processing: {e}")
            self.db_connection.rollback()
            import traceback
            traceback.print_exc()

    def setup_sample_data(self):
        """Setup static data for the video session"""
        try:
            # Static data for the entire video
            queries = [
                """INSERT IGNORE INTO location_dimension 
                   (location_key, metro_city_province, district, neighborhood, spot) 
                   VALUES 
                   (2, 'Islamabad', 'F-8', 'Margalla Road', 'Signal F-8/4'),
                   (3, 'Islamabad', 'F-10', 'Margalla Avenue', 'Signal F-10/2')""",
                
                """INSERT IGNORE INTO road_character_dimension 
                   (road_key, road_type, road_feature) 
                   VALUES 
                   (2, 'major_arterial', 'signalized_intersection'),
                   (3, 'collector_road', 'pedestrian_crossing')""",
                
                """INSERT IGNORE INTO behavior_feature 
                   (behavior_id, behavior_feature) 
                   VALUES 
                   (1, 'vehicle_acceleration'),
                   (2, 'vehicle_speed_5m'),
                   (3, 'vehicle_speed_10m'),
                   (4, 'vehicle_speed_15m'),
                   (5, 'vehicle_avg_speed'),
                   (6, 'vehicle_class'),
                   (10, 'pedestrian_parking'),
                   (11, 'pedestrian_direction'),
                   (12, 'pedestrian_speed'),
                   (13, 'pedestrian_vehicle_type'),
                   (14, 'pedestrian_psm'),
                   (15, 'pedestrian_parking')"""
            ]
            
            for query in queries:
                try:
                    self.db_cursor.execute(query)
                    print(f"Successfully executed query: {query[:100]}...")
                except Error as e:
                    print(f"Error executing query: {e}")
                    raise
                
            self.db_connection.commit()
            print("Static video data setup complete")
            
        except Error as e:
            print(f"Error setting up static data: {e}")
            if self.db_connection:
                self.db_connection.rollback()

def main():
    try:
        # Configuration
        model_path = "best(3).pt"
        pixels_per_meter = 100

        # Get video path from user
        while True:
            video_path = input("\nEnter the path to the MP4 file (or press Enter for default 'real4.mp4'): ").strip()
            if not video_path:
                video_path = "real4.mp4"
            if os.path.exists(video_path):
                print(f"Using video file: {video_path}")
                break
            print(f"Error: Video file not found at: {video_path}")

        # Get XML path from user
        while True:
            xml_path = input("\nEnter the path to the XML file (or press Enter for default 'D:\\fydp final\\ijp\\C0043M01.XML'): ").strip()
            if not xml_path:
                xml_path = "D:\\fydp final\\ijp\\C0043M01.XML"
            if os.path.exists(xml_path):
                print(f"Using XML file: {xml_path}")
                break
            print(f"Error: XML file not found at: {xml_path}")

        # Update the read_time_from_xml function with the provided XML path
        global read_time_from_xml
        old_read_time_from_xml = read_time_from_xml
        def new_read_time_from_xml(frame_count=0, fps=25):
            return old_read_time_from_xml(xml_path, frame_count, fps)
        read_time_from_xml = new_read_time_from_xml

        # Verify XML time is being used
        xml_time = read_time_from_xml()
        print(f"\nXML time verification: {xml_time}")
        print(f"System time for comparison: {datetime.now()}")
        
        # Processor selection with default values
        processors = {
            'PS': True,      # Pedestrian Speed Detector
            'PCA': True,     # Pedestrian Crossing Analyzer
            'CARPSM': True,  # Car PSM Detector
            'VCOMP': True,   # Vehicle Composition Analyzer
            'Ultralytics': True,  # Ultralytics Processor
            'BirdEye': True  # Bird's Eye View
        }

        print("\nCurrent processor settings (all enabled by default):")
        for name, enabled in processors.items():
            print(f"{name}: {'Enabled' if enabled else 'Disabled'}")

        change = input("\nWould you like to change any processor settings? (y/n): ").lower().strip()
        if change.startswith('y'):
            print("\nEnter 'y' to enable or 'n' to disable each processor:")
            for name in processors.keys():
                while True:
                    response = input(f"Enable {name}? (y/n): ").lower().strip()
                    if response in ['y', 'n']:
                        processors[name] = (response == 'y')
                        break
                    print("Please enter 'y' or 'n'")

            print("\nUpdated processor settings:")
            for name, enabled in processors.items():
                print(f"{name}: {'Enabled' if enabled else 'Disabled'}")
        
        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = PedestrianPipeline(model_path, pixels_per_meter, processors)
        
        # Process video
        print(f"\nProcessing video: {video_path}")
        pipeline.process_video(video_path)
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'pipeline' in locals():
            pipeline.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
