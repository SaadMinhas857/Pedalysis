import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time
import os
import csv
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import signal
import sys
import atexit
import xml.etree.ElementTree as ET

# Add DB configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'Traffic1',
    'password': 'pedestrian',
    'database': 'traffic_db'
}

class VehicleAnalyzer:
    def __init__(self, csv_filename: str, xml_path: str = 'D:\\fydp final\\ijp\\C0043M01.XML'):
        """Initialize Vehicle Analyzer"""
        try:
            # Store CSV filename and XML path
            self.csv_filename = csv_filename
            self.xml_path = xml_path
            print(f"Initializing VehicleAnalyzer with CSV file: {csv_filename}")
            
            # Initialize database connection as None - will be set by pipeline
            self.db_connection = None
            self.db_cursor = None
            
            # Parse XML file for metadata
            self.parse_xml_metadata()
            
            # Flag to track if data has been saved
            self.data_saved = False
            
            # Register cleanup handlers
            atexit.register(self.cleanup_handler)
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
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
            
            # Initialize time-based counting for 10-minute intervals
            self.interval_counts = []
            self.current_interval = 0
            self.current_interval_counts = {k: set() for k in self.vehicle_counts.keys()}
            
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

    def set_database_connection(self, db_connection, db_cursor):
        """Set database connection and cursor"""
        try:
            self.db_connection = db_connection
            self.db_cursor = db_cursor
            
            if self.db_connection and self.db_cursor:
                # Test the connection
                self.db_cursor.execute("SELECT 1")
                print("Database connection successfully set")
                return True
            else:
                print("Warning: Invalid database connection provided")
                return False
                
        except Error as e:
            print(f"Error setting database connection: {e}")
            self.db_connection = None
            self.db_cursor = None
            return False

    def parse_xml_metadata(self):
        """Parse XML file for metadata"""
        try:
            if os.path.exists(self.xml_path):
                tree = ET.parse(self.xml_path)
                root = tree.getroot()
                
                # Get creation date
                creation_date = root.find('.//{*}CreationDate')
                if creation_date is not None and 'value' in creation_date.attrib:
                    self.start_time = datetime.strptime(
                        creation_date.get('value').split('+')[0],
                        '%Y-%m-%dT%H:%M:%S'
                    )
                else:
                    self.start_time = datetime.now()
                    print("Warning: Using system time as XML creation date not found")
            else:
                self.start_time = datetime.now()
                print(f"Warning: XML file not found at {self.xml_path}, using system time")
                
        except Exception as e:
            print(f"Error parsing XML: {e}")
            self.start_time = datetime.now()

    def get_timestamp_from_frame(self, frame_count: int, fps: float = 25.0) -> datetime:
        """Calculate timestamp from frame count using XML start time"""
        seconds_offset = frame_count / fps
        return self.start_time + timedelta(seconds=seconds_offset)

    def insert_vehicle_data(self, track_id: int, vehicle_class: str, frame_time: float):
        """Insert vehicle traffic data into database"""
        if not self.ensure_db_connection():
            print("Cannot insert data: Database connection unavailable")
            return False

        try:
            # Start transaction
            self.db_cursor.execute("START TRANSACTION")
            
            # Calculate timestamp from frame time using XML time
            timestamp = self.get_timestamp_from_frame(int(frame_time * 25))  # Assuming 25 fps
            
            # Convert vehicle class to proper type
            vehicle_class = str(vehicle_class)
            track_id = int(track_id)
            
            # Get counts for each vehicle type
            vehicle_counts = {
                'biker': 1 if vehicle_class == '0' else 0,
                'bus': 1 if vehicle_class == '1' else 0,
                'motobike': 1 if vehicle_class == '2' else 0,
                'sedan': 1 if vehicle_class == '4' else 0,
                'taxi': 1 if vehicle_class == '5' else 0,
                'truck': 1 if vehicle_class == '6' else 0
            }
            
            # Insert time dimension with IGNORE to handle duplicates
            self.db_cursor.execute(
                """INSERT IGNORE INTO time_dimension 
                   (time_key, week, day, day_night, date, hour, minute)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    track_id,
                    f"Week{timestamp.strftime('%V')}",
                    timestamp.strftime('%A'),
                    'Day' if 6 <= timestamp.hour <= 18 else 'Night',
                    timestamp.date(),
                    timestamp.hour,
                    timestamp.minute
                )
            )
            
            # Insert vehicle traffic data
            self.db_cursor.execute(
                """INSERT INTO vehicle_traffic
                   (time_key, pedestrian_count, car_count, bus_count, truck_count)
                   VALUES (%s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                   pedestrian_count = pedestrian_count + VALUES(pedestrian_count),
                   car_count = car_count + VALUES(car_count),
                   bus_count = bus_count + VALUES(bus_count),
                   truck_count = truck_count + VALUES(truck_count)""",
                (
                    track_id,
                    vehicle_counts['biker'] + vehicle_counts['motobike'],  # Pedestrian count (bikers + motobikes)
                    vehicle_counts['sedan'] + vehicle_counts['taxi'],      # Car count (sedans + taxis)
                    vehicle_counts['bus'],                                 # Bus count
                    vehicle_counts['truck']                               # Truck count
                )
            )
            
            # Commit transaction
            self.db_connection.commit()
            print(f"Successfully inserted traffic data for vehicle {track_id} (class {vehicle_class})")
            return True
            
        except Error as e:
            print(f"Error inserting vehicle data: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False

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
            
            # Check if a 10-minute interval has passed
            current_interval = int((frame_time // 60) // 10)  # Convert to 10-minute intervals
            if current_interval > self.current_interval and current_interval <= 5:  # Max 6 intervals per hour
                # Save the previous interval's counts
                if sum(len(ids) for ids in self.current_interval_counts.values()) > 0:
                    self.interval_counts.append(self.current_interval_counts)
                    print(f"Saved interval {self.current_interval} with counts: {self.current_interval_counts}")
                # Reset for new interval
                self.current_interval_counts = {k: set() for k in self.vehicle_counts.keys()}
                self.current_interval = current_interval
            
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
                        self.current_interval_counts[vehicle_class].add(track_id)
                        print(f"Added vehicle {track_id} of class {vehicle_class} to interval {self.current_interval}")
                    
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
            
            # Save data every 10 minutes
            if frame_time % 600 == 0:  # Every 10 minutes (600 seconds)
                self.save_partial_data()
                
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
            cv2.putText(frame, f"Interval {self.current_interval + 1} Counts:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for vehicle_class, ids in self.current_interval_counts.items():
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
                
                # Write per 10-minute interval statistics
                for interval, counts in enumerate(self.interval_counts):
                    total = sum(len(ids) for ids in counts.values())
                    start_minute = interval * 10
                    end_minute = (interval + 1) * 10
                    
                    for vehicle_class, ids in counts.items():
                        count = len(ids)
                        percentage = (count / total * 100) if total > 0 else 0
                        writer.writerow([f"{start_minute:02d}:00 - {end_minute:02d}:00", 
                                       vehicle_class, 
                                       count, 
                                       f"{percentage:.2f}%"])
                    
                    # Add separator between intervals
                    writer.writerow([])
                
                # Write current interval if not empty
                total_current = sum(len(ids) for ids in self.current_interval_counts.values())
                if total_current > 0:
                    start_minute = self.current_interval * 10
                    end_minute = (self.current_interval + 1) * 10
                    for vehicle_class, ids in self.current_interval_counts.items():
                        count = len(ids)
                        percentage = (count / total_current * 100) if total_current > 0 else 0
                        writer.writerow([f"{start_minute:02d}:00 - {end_minute:02d}:00", 
                                       vehicle_class, 
                                       count, 
                                       f"{percentage:.2f}%"])
            
            print(f"\nResults saved to: {self.csv_filename}")
            
        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")

    def print_statistics(self):
        """Print final statistics"""
        print("\nVehicle Detection Statistics by 10-Minute Interval")
        print("=============================================")
        
        for interval, counts in enumerate(self.interval_counts):
            total = sum(len(ids) for ids in counts.values())
            start_minute = interval * 10
            end_minute = (interval + 1) * 10
            print(f"\nInterval {interval + 1} ({start_minute:02d}:00 - {end_minute:02d}:00):")
            print(f"Total Vehicles: {total}")
            print("Vehicle Class Distribution:")
            
            for vehicle_class, ids in counts.items():
                count = len(ids)
                percentage = (count / total * 100) if total > 0 else 0
                print(f"{vehicle_class}: {count} ({percentage:.1f}%)")
        
        # Print current interval if not empty
        total_current = sum(len(ids) for ids in self.current_interval_counts.values())
        if total_current > 0:
            start_minute = self.current_interval * 10
            end_minute = (self.current_interval + 1) * 10
            print(f"\nCurrent Interval ({start_minute:02d}:00 - {end_minute:02d}:00):")
            print(f"Total Vehicles: {total_current}")
            print("Vehicle Class Distribution:")
            
            for vehicle_class, ids in self.current_interval_counts.items():
                count = len(ids)
                percentage = (count / total_current * 100) if total_current > 0 else 0
                print(f"{vehicle_class}: {count} ({percentage:.1f}%)")

    def cleanup_handler(self):
        """Handle cleanup when the program exits"""
        if not self.data_saved:
            print("\nSaving partial vehicle count data...")
            self.save_partial_data()
            self.data_saved = True

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print(f"\nReceived signal {signum}. Saving partial data before exit...")
        self.save_partial_data()
        self.data_saved = True
        sys.exit(0)

    def save_partial_data(self):
        """Save interval-based vehicle count data to database"""
        try:
            # Save current interval data if any exists
            if sum(len(ids) for ids in self.current_interval_counts.values()) > 0:
                self.interval_counts.append(self.current_interval_counts)
                self.current_interval_counts = {k: set() for k in self.vehicle_counts.keys()}

            # Start transaction
            if not self.ensure_db_connection():
                print("Database connection not available")
                return False

            self.db_cursor.execute("START TRANSACTION")
            
            # Process each interval's data
            for interval, counts in enumerate(self.interval_counts):
                try:
                    # Calculate interval start time based on XML start time
                    interval_start = self.start_time + timedelta(minutes=interval * 10)
                    
                    # Format time key as YYYYMMDDHHMM
                    time_key = interval_start.strftime('%Y%m%d%H%M')
                    
                    # Calculate vehicle type counts for this interval
                    vehicle_type_counts = {
                        'pedestrian': len(counts.get('biker', set())) + len(counts.get('motobike', set())),
                        'car': len(counts.get('sedan', set())) + len(counts.get('taxi', set())),
                        'bus': len(counts.get('bus', set())),
                        'truck': len(counts.get('truck', set()))
                    }

                    print(f"Saving data for interval {interval} at time {interval_start}")
                    print(f"Vehicle counts: {vehicle_type_counts}")
                    
                    # Insert time dimension first
                    self.db_cursor.execute(
                        """INSERT IGNORE INTO time_dimension 
                           (time_key, week, day, day_night, date, hour, minute)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (
                            time_key,
                            f"Week{interval_start.strftime('%V')}",
                            interval_start.strftime('%A'),
                            'Day' if 6 <= interval_start.hour <= 18 else 'Night',
                            interval_start.date(),
                            interval_start.hour,
                            interval_start.minute
                        )
                    )
                    
                    # Insert vehicle traffic data
                    self.db_cursor.execute(
                        """INSERT INTO vehicle_traffic
                           (time_key, pedestrian_count, car_count, bus_count, truck_count)
                           VALUES (%s, %s, %s, %s, %s)
                           ON DUPLICATE KEY UPDATE
                           pedestrian_count = VALUES(pedestrian_count),
                           car_count = VALUES(car_count),
                           bus_count = VALUES(bus_count),
                           truck_count = VALUES(truck_count)""",
                        (
                            time_key,
                            vehicle_type_counts['pedestrian'],
                            vehicle_type_counts['car'],
                            vehicle_type_counts['bus'],
                            vehicle_type_counts['truck']
                        )
                    )
                    
                    print(f"Successfully saved data for interval {interval}")
                    
                except Error as e:
                    print(f"Error processing interval {interval}: {e}")
                    continue
            
            # Commit all changes
            self.db_connection.commit()
            print("All interval data saved successfully")
            
            # Clear processed intervals
            self.interval_counts = []
            
            # Also save to CSV as backup
            self.save_results_to_csv()
            
        except Error as e:
            print(f"Database error while saving interval data: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False
        except Exception as e:
            print(f"Unexpected error while saving interval data: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False
        
        return True

    def ensure_db_connection(self):
        """Ensure database connection is active"""
        if not self.db_connection or not self.db_cursor:
            print("Database connection not available")
            return False
            
        try:
            # Test the connection
            self.db_cursor.execute("SELECT 1")
            return True
        except Error as e:
            print(f"Database connection error: {e}")
            return False