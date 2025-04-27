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
                # Setup tables after connection is established
                self.setup_database_tables()
                return True
            else:
                print("Warning: Invalid database connection provided")
                return False
                
        except Error as e:
            print(f"Error setting database connection: {e}")
            self.db_connection = None
            self.db_cursor = None
            return False

    def setup_database_tables(self):
        """Setup necessary database tables"""
        try:
            print("\n=== Setting up database tables ===")
            
            # Create vehicle_traffic table with hour and minute_interval as primary key
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS vehicle_traffic (
                hour INT,
                minute_interval INT,  -- 0-5 representing 10-minute intervals
                pedestrian_count INT DEFAULT 0,
                car_count INT DEFAULT 0,
                bus_count INT DEFAULT 0,
                truck_count INT DEFAULT 0,
                PRIMARY KEY (hour, minute_interval)
            );
            """
            
            print(f"SQL Query:\n{create_table_sql}")
            
            # Execute the query and handle multiple result sets properly
            try:
                self.db_cursor.execute(create_table_sql)
                # Consume any remaining result sets
                while self.db_cursor.nextset() is not None:
                    pass
                print("✓ Successfully created vehicle_traffic table")
            except Error as e:
                print(f"❌ Error executing CREATE TABLE: {e}")
                raise
            
            # Commit the changes
            self.db_connection.commit()
            print("✓ Changes committed successfully")
            print("=== Database setup complete ===\n")
            
        except Error as e:
            print(f"\n❌ Error setting up vehicle_traffic table: {e}")
            print(f"Error code: {e.errno}")
            print(f"SQL State: {e.sqlstate}")
            if self.db_connection:
                self.db_connection.rollback()
                print("Changes rolled back")
            raise

    def parse_xml_metadata(self):
        """Parse XML file for metadata"""
        try:
            if os.path.exists(self.xml_path):
                tree = ET.parse(self.xml_path)
                root = tree.getroot()
                
                # Get creation date
                creation_date = root.find('.//{*}CreationDate')
                if creation_date is not None and 'value' in creation_date.attrib:
                    timestamp_str = creation_date.get('value').split('+')[0]
                    print(f"Found XML timestamp: {timestamp_str}")
                    self.start_time = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
                    print(f"Parsed XML start time: {self.start_time}")
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
        timestamp = self.start_time + timedelta(seconds=seconds_offset)
        print(f"Frame {frame_count} timestamp: {timestamp}")
        return timestamp

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
            
            # Calculate 10-minute interval (0-5)
            minute_interval = min(timestamp.minute // 10, 5)
            
            print(f"\nProcessing vehicle data:")
            print(f"XML timestamp: {timestamp}")
            print(f"Hour: {timestamp.hour}")
            print(f"Minute interval: {minute_interval} ({minute_interval*10:02d}-{min(minute_interval*10+9, 59):02d} minutes)")
            
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
            
            # Insert vehicle traffic data using hour and minute_interval
            insert_sql = """
            INSERT INTO vehicle_traffic
            (hour, minute_interval, pedestrian_count, car_count, bus_count, truck_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            pedestrian_count = pedestrian_count + VALUES(pedestrian_count),
            car_count = car_count + VALUES(car_count),
            bus_count = bus_count + VALUES(bus_count),
            truck_count = truck_count + VALUES(truck_count)
            """
            
            values = (
                timestamp.hour,
                minute_interval,
                vehicle_counts['biker'] + vehicle_counts['motobike'],  # Pedestrian count
                vehicle_counts['sedan'] + vehicle_counts['taxi'],      # Car count
                vehicle_counts['bus'],                                 # Bus count
                vehicle_counts['truck']                               # Truck count
            )
            
            print(f"\nExecuting SQL insert:")
            print(f"SQL: {insert_sql}")
            print(f"Values: {values}")
            
            self.db_cursor.execute(insert_sql, values)
            
            # Commit transaction
            self.db_connection.commit()
            print(f"✓ Successfully inserted traffic data for vehicle {track_id} (class {vehicle_class})")
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
            
            # Save data if interval changed or significant time passed
            if (current_interval > self.current_interval) or (frame_time % 300 == 0):  # Save every 5 minutes
                if sum(len(ids) for ids in self.current_interval_counts.values()) > 0:
                    print(f"\nSaving data for interval {self.current_interval}")
                    print(f"Current counts: {self.current_interval_counts}")
                    # Create a copy of current counts before saving
                    interval_data = {k: set(v) for k, v in self.current_interval_counts.items()}
                    self.interval_counts.append(interval_data)
                    self.save_partial_data()
                
                if current_interval > self.current_interval:
                    # Reset for new interval only if interval changed
                    self.current_interval_counts = {k: set() for k in self.vehicle_counts.keys()}
                    self.current_interval = current_interval
            
            # Process tracked objects and update counts
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
                    
                    # Draw bounding box and label
                    color = self.get_color_for_class(vehicle_class)
                    cv2.rectangle(processed_frame, (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])), color, 2)
                    
                    label = f"{vehicle_class} ID:{track_id}"
                    cv2.putText(processed_frame, label, (int(bbox[0]), int(bbox[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    frame_counts[vehicle_class] += 1
            
            # Draw counting region if defined
            if self.count_region is not None:
                cv2.polylines(processed_frame, [self.count_region], True, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Counting Region", 
                           tuple(self.count_region[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add count overlay
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
        try:
            print("\n=== Starting cleanup process ===")
            print("Video stopped. Saving final vehicle count data...")
            
            # Force save all remaining data
            if sum(len(ids) for ids in self.current_interval_counts.values()) > 0:
                # Get current XML time
                current_interval_start = self.start_time + timedelta(minutes=self.current_interval * 10)
                # Calculate proper 10-minute interval (0-5)
                minute_interval = min(current_interval_start.minute // 10, 5)
                
                # Calculate final counts
                vehicle_type_counts = {
                    'pedestrian': len(self.current_interval_counts.get('biker', set())) + 
                                 len(self.current_interval_counts.get('motobike', set())),
                    'car': len(self.current_interval_counts.get('sedan', set())) + 
                           len(self.current_interval_counts.get('taxi', set())),
                    'bus': len(self.current_interval_counts.get('bus', set())),
                    'truck': len(self.current_interval_counts.get('truck', set()))
                }
                
                print(f"\nFinal vehicle counts at time {current_interval_start}:")
                print(f"Current interval: {minute_interval}")
                print(f"XML time: {current_interval_start.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"10-minute interval: {minute_interval} ({minute_interval*10:02d}-{min(minute_interval*10+9, 59):02d} minutes)")
                print("Vehicle counts:", vehicle_type_counts)

                if not self.ensure_db_connection():
                    print("\nAttempting final database reconnection...")
                    try:
                        self.db_connection = mysql.connector.connect(**DB_CONFIG)
                        self.db_cursor = self.db_connection.cursor()
                        print("✓ Successfully reconnected for final save")
                    except Error as e:
                        print(f"❌ Failed to reconnect for final save: {e}")
                        return
                
                try:
                    print("\nStarting final database transaction...")
                    # Start final transaction
                    self.db_cursor.execute("START TRANSACTION")
                    
                    # Insert final vehicle traffic data
                    insert_sql = """
                        INSERT INTO vehicle_traffic
                        (hour, minute_interval, pedestrian_count, car_count, bus_count, truck_count)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        pedestrian_count = pedestrian_count + VALUES(pedestrian_count),
                        car_count = car_count + VALUES(car_count),
                        bus_count = bus_count + VALUES(bus_count),
                        truck_count = truck_count + VALUES(truck_count)
                    """
                    
                    values = (
                        current_interval_start.hour,
                        minute_interval,
                        vehicle_type_counts['pedestrian'],
                        vehicle_type_counts['car'],
                        vehicle_type_counts['bus'],
                        vehicle_type_counts['truck']
                    )
                    
                    print("\nExecuting SQL insert:")
                    print(f"SQL: {insert_sql}")
                    print(f"Values: {values}")
                    
                    self.db_cursor.execute(insert_sql, values)
                    
                    # Commit final transaction
                    self.db_connection.commit()
                    print("✓ Final vehicle count data saved successfully")
                    
                except Error as e:
                    print(f"\n❌ Error saving final data: {e}")
                    print(f"Error code: {e.errno}")
                    print(f"SQL State: {e.sqlstate}")
                    if self.db_connection:
                        self.db_connection.rollback()
                        print("Changes rolled back")
                finally:
                    print("\nPerforming final cleanup...")
                    # Save to CSV as backup
                    self.save_results_to_csv()
                    print("✓ CSV backup saved")
                    
                    # Close database connections
                    if self.db_cursor:
                        self.db_cursor.close()
                        print("✓ Database cursor closed")
                    if self.db_connection:
                        self.db_connection.close()
            
            self.data_saved = True
            print("\n=== Cleanup complete ===")
            
        except Exception as e:
            print(f"\n❌ Error during cleanup: {e}")
            import traceback
            print("Traceback:")
            traceback.print_exc()

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print(f"\nReceived signal {signum}. Performing emergency save...")
        self.cleanup_handler()  # Use the same cleanup logic
        sys.exit(0)

    def save_partial_data(self):
        """Save interval-based vehicle count data to database"""
        try:
            print("\n=== Starting partial data save ===")
            if not self.ensure_db_connection():
                print("Attempting to reconnect to database...")
                try:
                    self.db_connection = mysql.connector.connect(**DB_CONFIG)
                    self.db_cursor = self.db_connection.cursor()
                    print("✓ Successfully reconnected to database")
                except Error as e:
                    print(f"❌ Failed to reconnect to database: {e}")
                    return False

            # Start transaction
            print("\nStarting database transaction...")
            self.db_cursor.execute("START TRANSACTION")
            
            # Process each interval's data
            for interval, counts in enumerate(self.interval_counts):
                try:
                    # Calculate interval start time based on XML start time
                    interval_start = self.start_time + timedelta(minutes=interval * 10)
                    # Calculate proper 10-minute interval (0-5)
                    minute_interval = min(interval_start.minute // 10, 5)
                    
                    # Calculate vehicle type counts for this interval
                    vehicle_type_counts = {
                        'pedestrian': len(counts.get('biker', set())) + len(counts.get('motobike', set())),
                        'car': len(counts.get('sedan', set())) + len(counts.get('taxi', set())),
                        'bus': len(counts.get('bus', set())),
                        'truck': len(counts.get('truck', set()))
                    }

                    print(f"\nProcessing interval {interval}:")
                    print(f"XML time: {interval_start.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"10-minute interval: {minute_interval} ({minute_interval*10:02d}-{min(minute_interval*10+9, 59):02d} minutes)")
                    print(f"Vehicle counts: {vehicle_type_counts}")
                    
                    # Insert vehicle traffic data
                    insert_sql = """
                        INSERT INTO vehicle_traffic
                        (hour, minute_interval, pedestrian_count, car_count, bus_count, truck_count)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        pedestrian_count = pedestrian_count + VALUES(pedestrian_count),
                        car_count = car_count + VALUES(car_count),
                        bus_count = bus_count + VALUES(bus_count),
                        truck_count = truck_count + VALUES(truck_count)
                    """
                    
                    values = (
                        interval_start.hour,
                        minute_interval,
                        vehicle_type_counts['pedestrian'],
                        vehicle_type_counts['car'],
                        vehicle_type_counts['bus'],
                        vehicle_type_counts['truck']
                    )
                    
                    print(f"\nExecuting SQL insert for interval {interval}:")
                    print(f"SQL: {insert_sql}")
                    print(f"Values: {values}")
                    
                    self.db_cursor.execute(insert_sql, values)
                    print(f"✓ Successfully saved data for interval {interval}")
                            
                except Error as e:
                    print(f"\n❌ Error processing interval {interval}: {e}")
                    print(f"Error code: {e.errno}")
                    print(f"SQL State: {e.sqlstate}")
                    if self.db_connection:
                        self.db_connection.rollback()
                        print("Changes rolled back")
                    continue
            
            # Commit all changes
            self.db_connection.commit()
            print("\n✓ All interval data saved successfully")
            
            # Clear processed intervals after successful save
            self.interval_counts = []
            print("✓ Interval counts cleared")
            
            # Also save to CSV as backup
            self.save_results_to_csv()
            print("✓ CSV backup saved")
            
            print("=== Partial data save complete ===\n")
            return True
            
        except Error as e:
            print(f"\n❌ Database error while saving interval data: {e}")
            print(f"Error code: {e.errno}")
            print(f"SQL State: {e.sqlstate}")
            if self.db_connection:
                self.db_connection.rollback()
                print("Changes rolled back")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error while saving interval data: {e}")
            import traceback
            print("Traceback:")
            traceback.print_exc()
            if self.db_connection:
                self.db_connection.rollback()
                print("Changes rolled back")
            return False

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