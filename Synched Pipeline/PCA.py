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
from collections import deque


class PedestrianCrossingAnalyzer:
    def __init__(self, csv_filename: str, num_lanes: int = 3):
        """Initialize Pedestrian Crossing Analyzer"""
        try:
            # Store number of lanes
            self.num_lanes = num_lanes
            print(f"Initializing analyzer with {num_lanes} lanes")
            
            # Store points and lines
            self.lanes = []  # Will store num_lanes lanes, each with 4 points
            self.lane_lines = []  # Will store 3 vertical lines per lane
            self.crossing_times = {}  # {track_id: {line_num: time}}
            self.processed_ids = set()  # Keep track of fully processed IDs
            
            # Add ID mapping for consistent IDs
            self.id_mapping = {}  # {track_id: persistent_id}
            self.next_persistent_id = 1  # Counter for generating new persistent IDs
            
            # Simple parking detection variables
            self.vehicle_tracking = {}  # {vehicle_id: {'positions': [], 'stationary_frames': 0}}
            self.parked_vehicles = set()  # Set of currently parked vehicle IDs
            self.parking_violations = set()  # Set of pedestrian IDs with parking violations
            self.stationary_threshold = 15  # Frames to consider a vehicle as parked
            self.movement_threshold = 2.0  # Pixels of movement to consider vehicle as moving
            
            # Store CSV filename and setup CSV
            self.csv_filename = csv_filename
            self.setup_csv_file()
            
            print("Crossing Analyzer initialization complete.")
            print(f"Parking detection settings:")
            print(f"- Stationary threshold: {self.stationary_threshold} frames")
            print(f"- Movement threshold: {self.movement_threshold} pixels")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def load_existing_ids(self):
        """Load existing IDs from CSV file to maintain consistency"""
        try:
            if os.path.exists(self.csv_filename):
                with open(self.csv_filename, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if row and row[0]:  # If row exists and has an ID
                            persistent_id = int(row[0])
                            self.next_persistent_id = max(self.next_persistent_id, persistent_id + 1)
        except Exception as e:
            print(f"Error loading existing IDs: {str(e)}")

    def get_persistent_id(self, track_id):
        """Get or create a persistent ID for a track"""
        if track_id not in self.id_mapping:
            self.id_mapping[track_id] = self.next_persistent_id
            self.next_persistent_id += 1
        return self.id_mapping[track_id]

    def setup_csv_file(self):
        """Setup CSV file with headers based on number of lanes"""
        try:
            # Ensure logs directory exists
            os.makedirs('logs', exist_ok=True)
            
            # Create headers
            headers = ['Pedestrian_ID']
            for lane in range(1, self.num_lanes + 1):
                for line in range(1, 4):
                    headers.append(f'Lane{lane}_Line{line}_Time')
            headers.append('Parking')
            
            # Write headers to CSV file
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            print(f"Created new CSV file: {self.csv_filename}")
            print(f"Headers: {headers}")
            
        except Exception as e:
            print(f"Error setting up CSV file: {e}")
            raise

    def detect_initial_direction(self, track_id, position):
        """Detect initial direction of pedestrian movement"""
        if track_id not in self.pedestrian_directions:
            # Check which lane the pedestrian starts in
            for lane_idx, lane_points in enumerate(self.lanes):
                polygon = np.array(lane_points, np.int32)
                if cv2.pointPolygonTest(polygon, position, False) >= 0:
                    initial_lane = lane_idx + 1  # Convert to 1-based lane number
                    # If starting in first lane, assume forward direction
                    # If starting in last lane, assume reverse direction
                    direction = 'forward' if initial_lane == 1 else 'reverse' if initial_lane == self.num_lanes else None
                    if direction:
                        self.pedestrian_directions[track_id] = {
                            'direction': direction,
                            'initial_lane': initial_lane
                        }
                        print(f"Detected direction for Pedestrian {track_id}: {direction} (Starting in Lane {initial_lane})")
                    break

    def update_crossing_csv(self, track_id: int, line_num: int, crossing_time: str):
        """Update CSV with new crossing time"""
        try:
            # Get persistent ID for this track
            persistent_id = self.get_persistent_id(track_id)
            
            print(f"\nUpdating CSV for pedestrian {persistent_id}")
            print(f"Line crossed: {line_num}, Time: {crossing_time}")
            
            # Read existing data
            data = {}
            if os.path.exists(self.csv_filename):
                with open(self.csv_filename, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ped_id = int(row['Pedestrian_ID'])
                        data[ped_id] = row
            
            # Create new row if pedestrian doesn't exist
            if persistent_id not in data:
                new_row = {'Pedestrian_ID': str(persistent_id)}
                # Initialize all time columns to empty string
                for lane in range(1, self.num_lanes + 1):
                    for line in range(1, 4):
                        new_row[f'Lane{lane}_Line{line}_Time'] = ''
                new_row['Parking'] = '0'
                data[persistent_id] = new_row
            
            # Calculate which column to update
            lane = (line_num // 3) + 1
            line_in_lane = (line_num % 3) + 1
            column_name = f'Lane{lane}_Line{line_in_lane}_Time'
            
            # Update the crossing time
            data[persistent_id][column_name] = crossing_time
            
            # Update parking status
            data[persistent_id]['Parking'] = '1' if len(self.parked_vehicles) > 0 else '0'
            
            print(f"Updating column {column_name} with time {crossing_time}")
            
            # Write all data back to CSV
            with open(self.csv_filename, 'w', newline='') as f:
                # Get headers from first row of data
                headers = ['Pedestrian_ID']
                for lane in range(1, self.num_lanes + 1):
                    for line in range(1, 4):
                        headers.append(f'Lane{lane}_Line{line}_Time')
                headers.append('Parking')
                
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
                # Write all rows sorted by pedestrian ID
                for ped_id in sorted(data.keys()):
                    writer.writerow(data[ped_id])
                    print(f"Wrote row for pedestrian {ped_id}: {data[ped_id]}")
            
            print(f"Successfully updated CSV for Pedestrian {persistent_id}")
            
        except Exception as e:
            print(f"Error updating CSV: {e}")
            import traceback
            traceback.print_exc()

    def select_points(self, frame):
        """Allow user to select points for the specified number of lanes"""
        points = []
        frame_copy = frame.copy()
        
        # Create and setup window properly
        cv2.namedWindow('Select Points', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Points', 1920, 1080)
        cv2.setWindowProperty('Select Points', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Define labels for all lanes
        lane_labels = []
        for lane in range(1, self.num_lanes + 1):
            lane_labels.append([
                f'Lane{lane} Top Left',
                f'Lane{lane} Top Right',
                f'Lane{lane} Bottom Right',
                f'Lane{lane} Bottom Left'
            ])
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                total_points_needed = self.num_lanes * 4
                if len(points) < total_points_needed:
                    points.append((x, y))
                    
                    # Create a fresh copy for display
                    frame_display = frame_copy.copy()
                    
                    # Draw all points and lanes
                    for i, point in enumerate(points):
                        # Draw point
                        cv2.circle(frame_display, point, 5, (0, 255, 0), -1)
                        
                        # Add label
                        lane_idx = i // 4
                        point_idx = i % 4
                        label = lane_labels[lane_idx][point_idx]
                        cv2.putText(frame_display, label, 
                                  (point[0]+10, point[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw completed lanes
                    for lane_idx in range(len(points) // 4):
                        if (lane_idx + 1) * 4 <= len(points):
                            lane_points = points[lane_idx*4:(lane_idx+1)*4]
                            pts = np.array(lane_points, np.int32)
                            cv2.polylines(frame_display, [pts], True, (0, 255, 0), 2)
                            
                            # Draw temporary lines for visualization
                            if len(lane_points) == 4:
                                # Get top and bottom lines of the lane
                                top_line = [lane_points[0], lane_points[1]]
                                bottom_line = [lane_points[3], lane_points[2]]
                                
                                # Draw three vertical lines
                                for t in [0.2, 0.5, 0.8]:
                                    # Calculate points
                                    top_x = int(top_line[0][0] + t * (top_line[1][0] - top_line[0][0]))
                                    top_y = int(top_line[0][1] + t * (top_line[1][1] - top_line[0][1]))
                                    bottom_x = int(bottom_line[0][0] + t * (bottom_line[1][0] - bottom_line[0][0]))
                                    bottom_y = int(bottom_line[0][1] + t * (bottom_line[1][1] - bottom_line[0][1]))
                                    
                                    cv2.line(frame_display, (top_x, top_y), (bottom_x, bottom_y), (255, 0, 0), 2)
                    
                    cv2.imshow('Select Points', frame_display)
        
        cv2.setMouseCallback('Select Points', mouse_callback)
        cv2.imshow('Select Points', frame_copy)
        
        print(f"\nSelect {self.num_lanes * 4} points in this order:")
        for lane_label_set in lane_labels:
            for label in lane_label_set:
                print(f"- {label}")
        
        while len(points) < self.num_lanes * 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Store lanes and calculate lines
        self.lanes = [points[i:i+4] for i in range(0, len(points), 4)]
        self.calculate_lane_lines()
        
        return points

    def calculate_lane_lines(self):
        """Calculate three vertical lines for each lane"""
        self.lane_lines = []
        
        # Process lanes in original order (first input = Lane 1)
        for lane_idx, lane_points in enumerate(self.lanes):
            # Get top and bottom lines of the lane
            top_line = [lane_points[0], lane_points[1]]  # Left to right
            bottom_line = [lane_points[3], lane_points[2]]  # Left to right
            
            # Calculate three vertical lines (start, middle, end)
            for t in [0.2, 0.5, 0.8]:
                # Calculate top point
                top_x = int(top_line[0][0] + t * (top_line[1][0] - top_line[0][0]))
                top_y = int(top_line[0][1] + t * (top_line[1][1] - top_line[0][1]))
                
                # Calculate bottom point
                bottom_x = int(bottom_line[0][0] + t * (bottom_line[1][0] - bottom_line[0][0]))
                bottom_y = int(bottom_line[0][1] + t * (bottom_line[1][1] - bottom_line[0][1]))
                
                # Store line points
                self.lane_lines.append(((top_x, top_y), (bottom_x, bottom_y)))
                
                # Debug print for line creation
                line_num = len(self.lane_lines) % 3
                print(f"Created line {line_num + 1} for lane {lane_idx + 1} at t={t}")
                print(f"Line points: Top({top_x}, {top_y}), Bottom({bottom_x}, {bottom_y})")

    def draw_lane_lines(self, frame, lane_points):
        """Draw three vertical lines for a lane and the lane boundaries"""
        # Draw lane boundaries first
        lane_points_array = np.array([lane_points], dtype=np.int32)
        cv2.polylines(frame, lane_points_array, True, (0, 255, 0), 2)
        
        # Get lane number based on points
        lane_idx = -1
        for i, lane in enumerate(self.lanes):
            if lane == lane_points:
                lane_idx = i
                break
        
        if lane_idx >= 0:
            # Draw three vertical lines for this lane
            for i in range(3):
                line_idx = lane_idx * 3 + i
                if line_idx < len(self.lane_lines):
                    line = self.lane_lines[line_idx]
                    # Draw line with lane number and line number
                    cv2.line(frame, line[0], line[1], (255, 0, 0), 2)
                    # Add label
                    mid_x = (line[0][0] + line[1][0]) // 2
                    mid_y = (line[0][1] + line[1][1]) // 2
                    cv2.putText(frame, f"L{lane_idx+1}_{i+1}", (mid_x, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate the squared length of the line segment
        line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
        
        if line_length_sq == 0:
            # If line segment is actually a point, return distance to that point
            return np.sqrt((x - x1)**2 + (y - y1)**2)
        
        # Calculate projection point parameter
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))
        
        # Calculate projection point
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # Return distance to projection point
        return np.sqrt((x - proj_x)**2 + (y - proj_y)**2)

    def check_line_crossing(self, point, frame_time):
        """Check if point crosses any of the lines"""
        threshold = 20  # Distance threshold for line crossing
        
        # Get current position
        x, y = point
        
        # Check each line
        for line_num, line in enumerate(self.lane_lines):
            distance = self.point_to_line_distance(point, line[0], line[1])
            
            if distance < threshold:
                # Calculate lane and line numbers
                actual_lane = (line_num // 3) + 1
                line_in_lane = (line_num % 3) + 1
                
                print(f"\nPossible line crossing detected:")
                print(f"Line {line_num} (Lane {actual_lane}, Line {line_in_lane})")
                print(f"Distance to line: {distance:.2f} pixels")
                print(f"Point: ({x}, {y})")
                
                return line_num
                
        return None

    def process_video(self, video_path):
        """Process video file"""
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
            
            # Create and setup window properly
            cv2.namedWindow('Pedestrian Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Pedestrian Analysis', 1920, 1080)
            cv2.setWindowProperty('Pedestrian Analysis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Read first frame for point selection
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")
            
            # Select points
            self.select_points(first_frame)
            
            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create output directory if it doesn't exist
            os.makedirs('output', exist_ok=True)
            
            # Create output video writer
            output_path = os.path.join('output', f"processed_{os.path.basename(video_path)}")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get current system time
                current_time = datetime.now().strftime('%I:%M:%S %p')  # 12-hour format with AM/PM
                
                # Process frame
                processed_frame = self.process_frame(frame, [], frame_count/fps)
                
                # Show current system time on frame
                cv2.putText(processed_frame, f"Time: {current_time}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                out.write(processed_frame)
                cv2.imshow('Pedestrian Analysis', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
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
            if hasattr(self, 'csv_file'):
                self.csv_file.close()

    def calculate_speed(self, track_id, current_pos, current_time):
        """Calculate speed between consecutive positions"""
        if track_id not in self.pedestrian_speeds:
            self.pedestrian_speeds[track_id] = {
                'last_pos': current_pos,
                'last_time': current_time,
                'speeds': [],
                'avg_speed': 0,
                'max_speed': 0
            }
            return
        
        data = self.pedestrian_speeds[track_id]
        if data['last_time'] == current_time:
            return
        
        # Calculate distance in pixels
        dx = current_pos[0] - data['last_pos'][0]
        dy = current_pos[1] - data['last_pos'][1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Calculate time difference in seconds
        time_diff = current_time - data['last_time']
        
        if time_diff > 0:
            # Calculate speed in pixels per second
            speed = distance / time_diff
            # Convert to km/h (assuming 100 pixels = 1 meter)
            speed_kmh = (speed / 100) * 3.6
            
            if speed_kmh > 0.5:  # Filter out very small movements
                data['speeds'].append(speed_kmh)
                data['avg_speed'] = np.mean(data['speeds'])
                data['max_speed'] = max(data['max_speed'], speed_kmh)
        
        # Update last position and time
        data['last_pos'] = current_pos
        data['last_time'] = current_time

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
        
        return {
            'crossing_times': self.crossing_times.get(persistent_id, {})
        }

    def draw_lines(self, frame):
        """Draw all crossing lines on frame"""
        for line in self.lane_lines:
            cv2.line(frame, line[0], line[1], (255, 0, 0), 2)

    def is_point_in_last_lane(self, point):
        """Check if a point is inside the last lane polygon"""
        if not self.lanes:
            print("No lanes defined")
            return False
            
        # Get the last lane (nth lane)
        last_lane_points = self.lanes[-1]  # Get last lane points
        
        try:
            # Convert points to numpy array and ensure integer type
            polygon = np.array(last_lane_points, np.int32)
            point = (int(point[0]), int(point[1]))
            
            # Check if point is inside the polygon
            result = cv2.pointPolygonTest(polygon, point, False) >= 0
            
            if result:
                print(f"Vehicle detected in lane {len(self.lanes)} (parking zone) at position {point}")
            
            return result
            
        except Exception as e:
            print(f"Error checking point in last lane: {str(e)}")
            return False

    def check_vehicle_movement(self, positions):
        """Check if a vehicle has moved significantly based on its recent positions"""
        if len(positions) < 2:  # Need at least 2 positions to check movement
            return False
            
        try:
            # Get the last two positions
            last_pos = positions[-1]
            prev_pos = positions[-2]
            
            # Calculate movement distance
            movement = np.sqrt((last_pos[0] - prev_pos[0])**2 + 
                             (last_pos[1] - prev_pos[1])**2)
            
            # Check if movement exceeds threshold
            is_moving = movement >= self.movement_threshold
            
            print(f"Vehicle movement: {movement:.2f} pixels (threshold: {self.movement_threshold})")
            print(f"Vehicle is {'moving' if is_moving else 'stationary'}")
            
            return is_moving
            
        except Exception as e:
            print(f"Error checking vehicle movement: {str(e)}")
            return False

    def is_vehicle_stationary(self, positions, current_time, vehicle_data):
        """Check if a vehicle is stationary based on its recent positions"""
        if len(positions) < 5:  # Need at least 5 positions for reliable check
            print(f"Not enough positions ({len(positions)}) to check if vehicle is stationary")
            return False
            
        try:
            # Calculate movement for last 5 frames
            positions_list = list(positions)
            recent_positions = positions_list[-5:]
            total_movement = 0
            
            for i in range(len(recent_positions) - 1):
                dist = np.sqrt((recent_positions[i+1][0] - recent_positions[i][0])**2 + 
                             (recent_positions[i+1][1] - recent_positions[i][1])**2)
                total_movement += dist
            
            # Calculate average movement per frame
            avg_movement = total_movement / (len(recent_positions) - 1)
            print(f"Average movement: {avg_movement:.2f} pixels (threshold: {self.movement_threshold})")
            
            # Check if vehicle is moving
            is_moving = avg_movement >= self.movement_threshold
            
            if is_moving:
                # Increment moving frames counter and reset stationary counter
                vehicle_data['moving_frames'] = vehicle_data.get('moving_frames', 0) + 1
                vehicle_data['stationary_frames'] = 0
                print(f"Vehicle is moving - moving_frames: {vehicle_data['moving_frames']}")
                
                # If vehicle has been moving for threshold frames, reset parking status
                if vehicle_data['moving_frames'] >= self.stationary_threshold:
                    if vehicle_data.get('is_parked', False):
                        print(f"Vehicle is now moving - no longer considered parked")
                    vehicle_data['is_parked'] = False
                    vehicle_data['stationary_start_time'] = None
                return False
            else:
                # Reset moving frames counter and increment stationary counter
                vehicle_data['moving_frames'] = 0
                vehicle_data['stationary_frames'] = vehicle_data.get('stationary_frames', 0) + 1
                
                # Initialize stationary start time if not set
                if 'stationary_start_time' not in vehicle_data:
                    vehicle_data['stationary_start_time'] = time.time()
                
                stationary_duration = time.time() - vehicle_data['stationary_start_time']
                print(f"Vehicle is stationary for {stationary_duration:.1f} seconds")
                print(f"Stationary frames: {vehicle_data['stationary_frames']}")
                
                # Mark as parked if stationary for minimum time and frames
                if (vehicle_data['stationary_frames'] >= self.stationary_threshold and 
                    stationary_duration >= self.stationary_threshold):
                    if not vehicle_data.get('is_parked', False):
                        print(f"Vehicle detected as parked - stationary for {stationary_duration:.1f} seconds")
                    vehicle_data['is_parked'] = True
                    return True
            
            return vehicle_data.get('is_parked', False)
            
        except Exception as e:
            print(f"Error checking if vehicle is stationary: {str(e)}")
            return False

    def check_parking_violations(self, frame_time, pedestrians, vehicles, current_time):
        """Check for parking violations in the last lane"""
        try:
            # Process vehicles first to update stationary vehicles
            current_stationary = set()
            
            print(f"\nChecking parking violations in lane {len(self.lanes)} (last lane)")
            
            # First, check if we have any vehicles to process
            if not isinstance(vehicles, (list, np.ndarray)) or len(vehicles) == 0:
                print("No vehicles detected in frame")
                return
            
            print(f"Processing {len(vehicles)} vehicles")
            
            for vehicle in vehicles:
                try:
                    if len(vehicle) < 5:  # Need at least basic data
                        print(f"Invalid vehicle data format: {vehicle}")
                        continue
                    
                    # Convert values to native Python types
                    vehicle_id = int(float(vehicle[4]))
                    vehicle_class = int(float(vehicle[5])) if len(vehicle) >= 6 else 0
                    
                    # Skip if it's a pedestrian (class 3)
                    if vehicle_class == 3:
                        continue
                    
                    # Convert coordinates to float
                    x1, y1, x2, y2 = map(float, vehicle[:4])
                    vehicle_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    # Check if vehicle is in the last lane
                    if self.is_point_in_last_lane(vehicle_center):
                        print(f"\nProcessing vehicle {vehicle_id} in last lane:")
                        print(f"- Class: {vehicle_class}")
                        print(f"- Position: {vehicle_center}")
                        
                        # Initialize vehicle tracking if not exists
                        if vehicle_id not in self.vehicle_tracking:
                            self.vehicle_tracking[vehicle_id] = {
                                'positions': [],
                                'stationary_frames': 0,
                                'class': vehicle_class
                            }
                            print(f"Initialized new vehicle tracking for ID {vehicle_id}")
                        
                        vehicle_data = self.vehicle_tracking[vehicle_id]
                        vehicle_data['positions'].append(vehicle_center)
                        vehicle_data['last_time'] = current_time
                        
                        # Check if vehicle is stationary
                        if self.is_vehicle_stationary(vehicle_data['positions'], current_time, vehicle_data):
                            print(f"Vehicle {vehicle_id} confirmed as stationary")
                            self.parked_vehicles.add(vehicle_id)
                            current_stationary.add(vehicle_id)
                        elif vehicle_id in self.parked_vehicles:
                            print(f"Vehicle {vehicle_id} is now moving - removing from parked vehicles")
                            self.parked_vehicles.remove(vehicle_id)
                
                except Exception as e:
                    print(f"Error processing vehicle: {str(e)}")
                    continue
            
            # Remove vehicles that are no longer in the frame
            vehicles_to_remove = set(self.parked_vehicles) - current_stationary
            for vehicle_id in vehicles_to_remove:
                print(f"\nRemoving vehicle {vehicle_id} (no longer in frame)")
                if vehicle_id in self.vehicle_tracking:
                    del self.vehicle_tracking[vehicle_id]
                if vehicle_id in self.parked_vehicles:
                    self.parked_vehicles.remove(vehicle_id)
            
            # Process pedestrians and update their parking status
            if pedestrians is not None:
                print(f"\nProcessing {len(pedestrians)} pedestrians")
                print(f"Current parked vehicles: {len(self.parked_vehicles)}")
                
                for ped in pedestrians:
                    try:
                        if len(ped) < 5:  # Need at least ID
                            continue
                        
                        ped_id = self.get_persistent_id(int(float(ped[4])))
                        
                        # Check for parking violations
                        if len(self.parked_vehicles) > 0:
                            if ped_id not in self.parking_violations:
                                print(f"Marking pedestrian {ped_id} for parking violation")
                                print(f"Parked vehicles: {self.parked_vehicles}")
                            self.parking_violations.add(ped_id)
                        else:
                            if ped_id in self.parking_violations:
                                print(f"Clearing parking violation for pedestrian {ped_id}")
                                self.parking_violations.remove(ped_id)
                    
                    except Exception as e:
                        print(f"Error processing pedestrian: {str(e)}")
                        continue
            
            # Print current status
            if len(self.parked_vehicles) > 0:
                print(f"\nCurrently parked vehicles: {self.parked_vehicles}")
                print(f"Current parking violations: {self.parking_violations}")
            
        except Exception as e:
            print(f"Error in parking detection: {e}")
            import traceback
            traceback.print_exc()

    def process_frame(self, frame, tracks, frame_time, vehicles=None, current_time=None):
        """Process a frame and return annotated frame"""
        processed_frame = frame.copy()
        
        # Draw all lanes and lines
        for lane_idx, lane_points in enumerate(self.lanes):
            self.draw_lane_lines(processed_frame, lane_points)
            
            # Highlight the last lane (parking zone)
            if lane_idx == len(self.lanes) - 1:
                overlay = processed_frame.copy()
                points = np.array(lane_points, np.int32)
                cv2.fillPoly(overlay, [points], (0, 255, 255))  # Yellow tint
                cv2.addWeighted(overlay, 0.2, processed_frame, 0.8, 0, processed_frame)
                cv2.polylines(processed_frame, [points], True, (0, 255, 255), 3)
                cv2.putText(processed_frame, "PARKING ZONE", 
                          (points[0][0], points[0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Update parking detection
        self.check_parking_violations(frame_time, tracks, vehicles, current_time)
        
        # Process pedestrian tracks for lane crossing
        for track in tracks:
            if len(track) < 5:
                continue
                
            track_id = int(track[4])
            persistent_id = self.get_persistent_id(track_id)
            bbox = track[:4]
            
            # Skip invalid boxes
            if any(x <= 0 for x in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
                
            # Get center point of bottom of bounding box
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int(bbox[3])  # Bottom of box
            
            # Process track for lane crossing
            crossing_info = self.process_track(track_id, (center_x, center_y), frame_time, current_time)
            
            # Draw pedestrian box
            color = (0, 0, 255) if persistent_id in self.parking_violations else (0, 255, 0)
            cv2.rectangle(processed_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Add labels
            labels = [f"ID:{persistent_id}"]
            if persistent_id in self.parking_violations:
                labels.append("PARKING VIOLATION")
            
            # Display all crossing times for this pedestrian
            if crossing_info and crossing_info['crossing_times']:
                for line_num, time in crossing_info['crossing_times'].items():
                    lane_num = (line_num // 3) + 1
                    line_in_lane = (line_num % 3) + 1
                    labels.append(f"L{lane_num}-{line_in_lane}: {time}")
            
            # Draw labels
            y_offset = int(bbox[1]) - 10
            for label in labels:
                cv2.putText(processed_frame, label,
                           (int(bbox[0]), y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset -= 20
        
        # Draw parked vehicles
        if vehicles is not None:
            for vehicle in vehicles:
                if len(vehicle) >= 5 and int(float(vehicle[4])) in self.parked_vehicles:
                    # Draw parked vehicle box
                    x1, y1, x2, y2 = map(int, map(float, vehicle[:4]))
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(processed_frame, f"PARKED #{int(float(vehicle[4]))}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add parking status
        cv2.putText(processed_frame, 
                   f"Parked Vehicles: {len(self.parked_vehicles)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        return processed_frame

    def update_parking_detection(self, frame_time, tracks, vehicles):
        """Simple parking detection logic"""
        try:
            # Reset parked vehicles each frame
            currently_tracked = set()
            
            # Process vehicles first
            if vehicles is not None:
                print(f"\nProcessing vehicles for parking detection:")
                print(f"Total vehicles detected: {len(vehicles)}")
                
                for vehicle in vehicles:
                    try:
                        # Check if we have the basic bbox and ID
                        if len(vehicle) < 5:
                            print(f"Skipping vehicle - missing basic data: {vehicle}")
                            continue
                        
                        # Convert coordinates and ID to proper format
                        x1, y1, x2, y2 = map(float, vehicle[:4])
                        vehicle_id = int(float(vehicle[4]))
                        vehicle_class = int(float(vehicle[5])) if len(vehicle) >= 6 else 0
                        
                        # Skip pedestrians (class 3)
                        if vehicle_class == 3:
                            continue
                        
                        # Get vehicle center point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        center = (center_x, center_y)
                        
                        # Check if vehicle is in last lane
                        if self.is_point_in_last_lane(center):
                            # Initialize or update tracking
                            if vehicle_id not in self.vehicle_tracking:
                                self.vehicle_tracking[vehicle_id] = {
                                    'positions': [],
                                    'stationary_frames': 0,
                                    'class': vehicle_class
                                }
                            
                            # Update positions
                            vehicle_data = self.vehicle_tracking[vehicle_id]
                            vehicle_data['positions'].append(center)
                            
                            # Check movement
                            if self.check_vehicle_movement(vehicle_data['positions']):
                                vehicle_data['stationary_frames'] = 0
                                if vehicle_id in self.parked_vehicles:
                                    self.parked_vehicles.remove(vehicle_id)
                            else:
                                vehicle_data['stationary_frames'] += 1
                                frames = vehicle_data['stationary_frames']
                                
                                if frames >= self.stationary_threshold:
                                    if vehicle_id not in self.parked_vehicles:
                                        self.parked_vehicles.add(vehicle_id)
                            
                            currently_tracked.add(vehicle_id)
                    
                    except Exception as e:
                        print(f"Error processing vehicle: {e}")
                        continue
            
            # Remove vehicles no longer tracked
            for vehicle_id in list(self.vehicle_tracking.keys()):
                if vehicle_id not in currently_tracked:
                    del self.vehicle_tracking[vehicle_id]
                    if vehicle_id in self.parked_vehicles:
                        self.parked_vehicles.remove(vehicle_id)
            
            # Process pedestrians
            if tracks is not None:
                for ped in tracks:
                    try:
                        if len(ped) < 5:
                            continue
                        
                        ped_id = self.get_persistent_id(int(float(ped[4])))
                        
                        # Check for parking violations
                        if len(self.parked_vehicles) > 0:
                            if ped_id not in self.parking_violations:
                                self.parking_violations.add(ped_id)
                        else:
                            if ped_id in self.parking_violations:
                                self.parking_violations.remove(ped_id)
                    
                    except Exception as e:
                        print(f"Error processing pedestrian: {e}")
                        continue
        
        except Exception as e:
            print(f"Error in parking detection: {e}")
            import traceback
            traceback.print_exc()

def main():
    try:
        # CHANGE THIS PATH to your video's location
        video_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\export6.mp4"
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        
        # Get number of lanes from user
        while True:
            try:
                num_lanes = int(input("Enter the number of lanes (minimum 1): "))
                if num_lanes >= 1:
                    break
                print("Please enter a number greater than or equal to 1")
            except ValueError:
                print("Please enter a valid number")
        
        # CHANGE THIS PATH to your CSV file's location
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"pedestrian_crossings_{timestamp}_{num_lanes}lanes.csv"
        
        analyzer = PedestrianCrossingAnalyzer(csv_filename, num_lanes)
        analyzer.process_video(video_path)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return

if __name__ == "__main__":
    main() 