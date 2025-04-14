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

# Add DB configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'Traffic1',
    'password': 'pedestrian',
    'database': 'traffic_db'
}

def read_time_from_xml(xml_file_path='D:\\fydp final\\ijp\\C0043M01.XML', frame_count=0, fps=25):
    """Read time from XML file and adjust by frame count"""
    try:
        # Check if XML file exists
        if not os.path.exists(xml_file_path):
            print(f"XML file not found: {xml_file_path}. Using system time instead.")
            return datetime.now()
        
        # Parse XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Find CreationDate element and get its value attribute
        creation_date = root.find('.//{*}CreationDate')
        if creation_date is None or 'value' not in creation_date.attrib:
            print("CreationDate not found in XML. Using system time instead.")
            return datetime.now()
        
        # Get the timestamp string from the value attribute
        timestamp_str = creation_date.get('value')
        
        # Parse the ISO 8601 format timestamp
        # Remove timezone info as it's not needed for our purposes
        timestamp_str = timestamp_str.split('+')[0]
        base_time = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
        
        # Calculate time offset based on frame count and FPS
        seconds_offset = frame_count / fps
        time_delta = timedelta(seconds=seconds_offset)
        xml_time = base_time + time_delta
        
        return xml_time
        
    except Exception as e:
        print(f"Error reading time from XML: {e}. Using system time instead.")
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
            os.makedirs('logs', exist_ok=True)
            
            # Create and store CSV paths with unique names
            self.ped_csv = f'logs/pedestrian_analysis_{timestamp}.csv'
            self.vehicle_csv = f'logs/vehicle_analysis_{timestamp}.csv'
            self.vcomp_csv = f'logs/vehicle_counts_{timestamp}.csv'
            self.ultra_csv = f'logs/ultralytics_analysis_{timestamp}.csv'
            
            print(f"CSV files will be saved to logs directory with timestamp: {timestamp}")
            
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
                """CREATE TABLE IF NOT EXISTS location_dimension (
                    location_key INT PRIMARY KEY,
                    metro_city_province VARCHAR(255),
                    district VARCHAR(255),
                    neighborhood VARCHAR(255),
                    spot VARCHAR(255)
                )""",
                
                """CREATE TABLE IF NOT EXISTS road_character_dimension (
                    road_key INT PRIMARY KEY,
                    road_type VARCHAR(255),
                    road_feature VARCHAR(255)
                )""",
                
                """CREATE TABLE IF NOT EXISTS time_dimension (
                    time_key INT PRIMARY KEY,
                    week VARCHAR(50),
                    day VARCHAR(50),
                    day_night VARCHAR(10),
                    hour VARCHAR(50),
                    full_timestamp DATETIME
                )""",
                
                """CREATE TABLE IF NOT EXISTS scene_dimension (
                    scene_key INT PRIMARY KEY,
                    object_type VARCHAR(50),
                    event_type VARCHAR(50)
                )""",
                
                """CREATE TABLE IF NOT EXISTS behavior_feature (
                    behavior_id INT PRIMARY KEY,
                    behavior_feature VARCHAR(255)
                )""",
                
                """CREATE TABLE IF NOT EXISTS fact_table (
                    time_key INT,
                    location_key INT,
                    road_key INT,
                    scene_key INT,
                    scene_ratio FLOAT,
                    video_code VARCHAR(255),
                    PRIMARY KEY (time_key, location_key, road_key, scene_key),
                    FOREIGN KEY (time_key) REFERENCES time_dimension(time_key),
                    FOREIGN KEY (location_key) REFERENCES location_dimension(location_key),
                    FOREIGN KEY (road_key) REFERENCES road_character_dimension(road_key),
                    FOREIGN KEY (scene_key) REFERENCES scene_dimension(scene_key)
                )""",
                
                """CREATE TABLE IF NOT EXISTS scene_behavior_feature (
                    scene_key INT,
                    object_type VARCHAR(50),
                    behavior_id INT,
                    behavior_value FLOAT,
                    PRIMARY KEY (scene_key, behavior_id),
                    FOREIGN KEY (scene_key) REFERENCES scene_dimension(scene_key),
                    FOREIGN KEY (behavior_id) REFERENCES behavior_feature(behavior_id)
                )"""
            ]
            
            for table in tables:
                self.db_cursor.execute(table)
            
            # Insert behavior features if they don't exist
            behaviors = [
                (1, 'Reference_Speed'),
                (2, 'Speed_5m'),
                (3, 'Speed_10m'),
                (4, 'Speed_15m'),
                (5, 'Average_Speed'),
                (6, 'Pedestrian_Speed'),
                (7, 'Vehicle_Class'),
                (8, 'Acceleration')  # New behavior ID for acceleration/deceleration
            ]
            
            for behavior in behaviors:
                self.db_cursor.execute(
                    "INSERT IGNORE INTO behavior_feature (behavior_id, behavior_feature) VALUES (%s, %s)",
                    behavior
                )
            
            # Insert sample location and road data
            self.db_cursor.execute(
                """INSERT IGNORE INTO location_dimension 
                   (location_key, metro_city_province, district, neighborhood, spot)
                   VALUES (1, 'Sample City', 'Sample District', 'Sample Area', 'Sample Spot')"""
            )
            
            self.db_cursor.execute(
                """INSERT IGNORE INTO road_character_dimension
                   (road_key, road_type, road_feature)
                   VALUES (1, 'Sample Road Type', 'Sample Feature')"""
            )
            
            self.db_connection.commit()
            print("Database tables setup complete")
            
        except Error as e:
            print(f"Error setting up database tables: {e}")
            raise

    def insert_track_data(self, track_id: int, object_type: str, speeds: dict = None, class_id: int = None):
        """Insert track data into database"""
        if not self.db_connection or not self.db_cursor:
            print("Warning: Database connection not available. Cannot insert track data.")
            return
            
        try:
            # Ensure all values are proper Python types
            track_id = int(track_id)
            current_time = read_time_from_xml()  # Use XML time instead of system time
            
            # Print debug information
            print(f"Inserting track data for {object_type} ID {track_id}")
            if speeds:
                print(f"Speeds to insert: {speeds}")
            
            # Insert time dimension
            self.db_cursor.execute(
                """INSERT IGNORE INTO time_dimension 
                   (time_key, week, day, day_night, hour, full_timestamp)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    track_id,
                    f"Week{current_time.strftime('%V')}",
                    current_time.strftime('%A'),
                    'Day' if 6 <= current_time.hour <= 18 else 'Night',
                    current_time.strftime('%Y-%m-%d %H:00:00'),
                    current_time.strftime('%Y-%m-%d %H:%M:%S')
                )
            )
            print("Time dimension inserted")
            
            # Insert scene dimension
            self.db_cursor.execute(
                """INSERT IGNORE INTO scene_dimension
                   (scene_key, object_type, event_type)
                   VALUES (%s, %s, %s)""",
                (track_id, str(object_type), 'movement')
            )
            print("Scene dimension inserted")
            
            # Insert fact table
            self.db_cursor.execute(
                """INSERT IGNORE INTO fact_table
                   (time_key, location_key, road_key, scene_key, scene_ratio, video_code)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (track_id, 1, 1, track_id, 1.0, f"video_{self.video_id}_{track_id}")
            )
            print("Fact table inserted")
            
            # Insert behavior features for speeds
            if speeds:
                for behavior_id, speed in speeds.items():
                    # Ensure proper types
                    behavior_id = int(behavior_id)
                    speed = float(speed)
                    
                    # Skip if speed is too low
                    if speed < 3.0:
                        print(f"Skipping speed {speed:.2f} km/h for behavior_id {behavior_id} (below threshold)")
                        continue
                    
                    print(f"Inserting speed {speed:.2f} km/h for behavior_id {behavior_id}")
                    
                    try:
                        # Insert or update the speed
                        self.db_cursor.execute(
                            """INSERT INTO scene_behavior_feature
                               (scene_key, object_type, behavior_id, behavior_value)
                               VALUES (%s, %s, %s, %s)
                               ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)""",
                            (track_id, str(object_type), behavior_id, speed)
                        )
                        print(f"Speed {speed:.2f} km/h inserted for behavior_id {behavior_id}")
                    except Error as e:
                        print(f"Error inserting speed: {e}")
                        print(f"Debug - track_id: {track_id} ({type(track_id)}), "
                              f"behavior_id: {behavior_id} ({type(behavior_id)}), "
                              f"speed: {speed} ({type(speed)})")
                        raise
            
            # Insert vehicle class if provided
            if class_id is not None:
                class_id = int(class_id)
                print(f"Inserting vehicle class {class_id} for track {track_id}")
                self.db_cursor.execute(
                    """INSERT INTO scene_behavior_feature
                       (scene_key, object_type, behavior_id, behavior_value)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)""",
                    (track_id, str(object_type), 7, float(class_id))
                )
                print("Vehicle class inserted")
            
            # Commit the transaction
            self.db_connection.commit()
            print(f"Successfully committed track data for {object_type} ID {track_id}")
            
        except Error as e:
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
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        try:
            print(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties and setup
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            if self.fps == 0:  # If FPS cannot be determined, use default
                self.fps = 25  # Default FPS
            print(f"Video FPS: {self.fps}")
            
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
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")
            
            # Setup bird's eye view
            target_width, target_height = self.setup_birds_eye_view(first_frame)
            
            # Select lanes and target y-coordinate for car detection if CARPSM is enabled
            if self.processors['PCA']:
                self.crossing_analyzer.select_points(first_frame)
                if self.processors['CARPSM']:
                    self.car_detector.lanes = self.crossing_analyzer.lanes
                    self.car_detector.select_target_y(first_frame)
            
            # Select reference points for Ultralytics processor if enabled
            if self.processors['Ultralytics']:
                self.ultralytics_processor.select_reference_points(first_frame)
            
            # Select vehicle counting region if VCOMP is enabled
            if self.processors['VCOMP']:
                print("\nSelect region for vehicle counting:")
                self.vehicle_analyzer.select_count_region(first_frame)
            
            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create output directory and video writers
            os.makedirs('output', exist_ok=True)
            
            # Initialize video writers only for enabled processors
            ped_output_path = os.path.join('output', f"processed_{os.path.basename(video_path)}")
            ped_out = cv2.VideoWriter(ped_output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            
            if self.processors['CARPSM']:
                car_output_path = os.path.join('output', f"carpsm_{os.path.basename(video_path)}")
                car_out = cv2.VideoWriter(car_output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            
            if self.processors['VCOMP']:
                vcomp_output_path = os.path.join('output', f"vcomp_{os.path.basename(video_path)}")
                vcomp_out = cv2.VideoWriter(vcomp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            
            if self.processors['BirdEye']:
                birds_eye_path = os.path.join('output', f"birds_eye_{os.path.basename(video_path)}")
                birds_eye_out = cv2.VideoWriter(birds_eye_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (target_width, target_height))
            
            if self.processors['Ultralytics']:
                ultra_output_path = os.path.join('output', f"ultralytics_{os.path.basename(video_path)}")
                ultra_out = cv2.VideoWriter(ultra_output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = read_time_from_xml(frame_count=frame_count, fps=self.fps).strftime('%H:%M:%S')
                frame_time = float(frame_count) / float(self.fps)
                
                try:
                    # Get detections from YOLO model with NMS
                    results = self.model(frame, conf=self.conf_threshold, iou=0.5)[0]
                    
                    # Process detections and get tracked objects
                    ped_tracks, vehicle_tracks = self.process_detections(results, frame)
                    
                    # Process frames through external components with tracked objects
                    ped_frame = frame.copy()
                    if self.processors['PCA']:
                        ped_frame = self.crossing_analyzer.process_frame(ped_frame, ped_tracks, frame_time, vehicle_tracks, current_time)
                    if self.processors['PS']:
                        ped_frame = self.speed_detector.process_frame(ped_frame, ped_tracks, frame_time)
                    
                    car_frame = frame.copy()
                    if self.processors['CARPSM']:
                        car_frame = self.car_detector.process_frame(car_frame, vehicle_tracks, current_time)
                    
                    vcomp_frame = frame.copy()
                    if self.processors['VCOMP']:
                        vcomp_frame = self.vehicle_analyzer.process_frame(vcomp_frame, vehicle_tracks, self.vehicle_classes, frame_time)
                    
                    # Process bird's eye view only if enabled
                    if self.processors['BirdEye']:
                        birds_eye_frame, visible_ids = self.process_birds_eye_view(frame, vehicle_tracks)
                    else:
                        birds_eye_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                        visible_ids = set()
                    
                    # Process frame with Ultralytics processor
                    ultra_frame = frame.copy()
                    if self.processors['Ultralytics']:
                        ultra_frame = self.ultralytics_processor.process_frame(ultra_frame, vehicle_tracks, visible_ids)
                    
                    # Add timestamp and frame info to frames
                    for frame_to_write in [ped_frame] + \
                                        ([car_frame] if self.processors['CARPSM'] else []) + \
                                        ([vcomp_frame] if self.processors['VCOMP'] else []) + \
                                        ([ultra_frame] if self.processors['Ultralytics'] else []):
                        cv2.putText(frame_to_write, f"Time: {current_time}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame_to_write, f"Frame: {frame_count} | FPS: {self.fps:.1f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Write frames only for enabled processors
                    ped_out.write(ped_frame)
                    if self.processors['CARPSM']:
                        car_out.write(car_frame)
                    if self.processors['VCOMP']:
                        vcomp_out.write(vcomp_frame)
                    if self.processors['BirdEye']:
                        birds_eye_out.write(birds_eye_frame)
                    if self.processors['Ultralytics']:
                        ultra_out.write(ultra_frame)
                    
                    # Display frames only for enabled processors
                    cv2.imshow('Traffic Analysis', ped_frame)
                    if self.processors['CARPSM']:
                        cv2.imshow('Car PSM', car_frame)
                    if self.processors['VCOMP']:
                        cv2.imshow('Vehicle Composition', vcomp_frame)
                    if self.processors['BirdEye']:
                        cv2.imshow('Bird\'s Eye View', birds_eye_frame)
                    if self.processors['Ultralytics']:
                        cv2.imshow('Ultralytics Processor', ultra_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing interrupted by user")
                        break
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    # Write original frame if processing fails
                    ped_out.write(frame)
                    if self.processors['CARPSM']:
                        car_out.write(frame)
                    if self.processors['VCOMP']:
                        vcomp_out.write(frame)
                    if self.processors['BirdEye']:
                        birds_eye_out.write(np.zeros((target_height, target_width, 3), dtype=np.uint8))
                    if self.processors['Ultralytics']:
                        ultra_out.write(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
            # Save final results
            print("Saving final results...")
            if self.processors['VCOMP']:
                self.vehicle_analyzer.save_results_to_csv()
                self.vehicle_analyzer.print_statistics()
            if self.processors['Ultralytics']:
                self.ultralytics_processor.save_results_to_csv()
                self.ultralytics_processor.print_statistics()
            
            print(f"\nProcessing complete!")
            print(f"Pedestrian output saved to: {ped_output_path}")
            if self.processors['CARPSM']:
                print(f"Car PSM output saved to: {car_output_path}")
            if self.processors['VCOMP']:
                print(f"Vehicle Composition output saved to: {vcomp_output_path}")
            if self.processors['BirdEye']:
                print(f"Bird's Eye View output saved to: {birds_eye_path}")
            if self.processors['Ultralytics']:
                print(f"Ultralytics Processor output saved to: {ultra_output_path}")
            
            print(f"CSV files saved to logs directory:")
            if self.processors['PS']:
                print(f"  - Pedestrian analysis: {os.path.abspath(self.ped_csv)}")
            if self.processors['CARPSM']:
                print(f"  - Vehicle analysis: {os.path.abspath(self.vehicle_csv)}")
            if self.processors['VCOMP']:
                print(f"  - Vehicle counts: {os.path.abspath(self.vcomp_csv)}")
            if self.processors['Ultralytics']:
                print(f"  - Ultralytics analysis: {os.path.abspath(self.ultra_csv)}")
            
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            raise
            
        finally:
            if 'cap' in locals():
                cap.release()
            if 'ped_out' in locals():
                ped_out.release()
            if 'car_out' in locals() and self.processors['CARPSM']:
                car_out.release()
            if 'vcomp_out' in locals() and self.processors['VCOMP']:
                vcomp_out.release()
            if 'birds_eye_out' in locals() and self.processors['BirdEye']:
                birds_eye_out.release()
            if 'ultra_out' in locals() and self.processors['Ultralytics']:
                ultra_out.release()
            cv2.destroyAllWindows()

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
                                speeds={8: float(accel_value)},  # Behavior ID 8 for acceleration
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

def main():
    try:
        # Configuration
        model_path = "best(3).pt"
        video_path = "export6.mp4" 
        pixels_per_meter = 100

        # Verify XML time is being used
        xml_time = read_time_from_xml()
        print(f"XML time verification: {xml_time}")
        print(f"System time for comparison: {datetime.now()}")
        
        # Processor selection
        print("\nSelect which processors to enable (y/n for each):")
        processors = {
            'PS': input("Enable Pedestrian Speed Detector (PS)? ").lower().startswith('y'),
            'PCA': input("Enable Pedestrian Crossing Analyzer (PCA)? ").lower().startswith('y'),
            'Ultralytics': input("Enable Ultralytics Processor? ").lower().startswith('y'),
            'VCOMP': input("Enable Vehicle Composition Analyzer (VCOMP)? ").lower().startswith('y'),
            'CARPSM': input("Enable Car PSM Detector (CARPSM)? ").lower().startswith('y'),
            'BirdEye': input("Enable Bird's Eye View? ").lower().startswith('y')
        }
        
        print("\nEnabled processors:")
        for name, enabled in processors.items():
            print(f"{name}: {'Enabled' if enabled else 'Disabled'}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        
        pipeline = PedestrianPipeline(model_path, pixels_per_meter, processors)
        pipeline.process_video(video_path)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return

if __name__ == "__main__":
    main()
