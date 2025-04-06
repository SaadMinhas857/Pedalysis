import os
import cv2
import pandas as pd
import time
import numpy as np
from datetime import datetime
import csv

# Define the is_near_line function here
def is_near_line(x, y, line_start, line_end, offset):
    """ Check if point (x, y) is near a line segment defined by line_start and line_end """
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    point = np.array([x, y])
    
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    # Vector from line_start to point
    point_vec = point - line_start
    
    # Project point_vec onto line_vec to get the closest point on the line
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        closest_point = line_start
    else:
        projection = np.dot(point_vec, line_vec) / line_len_sq
        if projection < 0:
            closest_point = line_start
        elif projection > 1:
            closest_point = line_end
        else:
            closest_point = line_start + projection * line_vec
    
    # Check if the distance from the point to the closest point on the line is within the offset
    distance = np.linalg.norm(point - closest_point)
    return distance <= offset

class VehicleSpeedDetector:
    def __init__(self, csv_filename: str):
        """Initialize Vehicle Speed Detector"""
        try:
            # Store CSV filename
            self.csv_filename = csv_filename
            print(f"Initializing VehicleSpeedDetector with CSV file: {csv_filename}")
            
            # Define line coordinates - will be set by user
            self.blue_line_start = None
            self.blue_line_end = None
            self.green_line_start = None
            self.green_line_end = None
            self.brown_line_start = None
            self.brown_line_end = None
            self.red_line_start = None
            self.red_line_end = None
            
            # Distances from blue line
            self.distance_red_blue = 15  # meters
            self.distance_green_blue = 30  # meters
            self.distance_brown_blue = 45  # meters
            
            self.offset = 8  # Increased offset for better line detection
            
            # Initialize tracking variables
            self.vehicle_tracking = {}  # {id: {'green': {'time': time, 'crossed': False}, ...}}
            self.vehicle_speeds = {}    # {id: {'green': speed, 'brown': speed, 'red': speed}}
            self.counter_down = []
            self.counter_up = []
            
            # Initialize CSV data
            self.csv_data = {'id': [], 'green': [], 'brown': [], 'red': []}
            
            # Setup CSV file
            self.setup_csv_file()
            
            print("VehicleSpeedDetector initialization complete.")
            
        except Exception as e:
            raise Exception(f"Initialization error: {str(e)}")
    
    def setup_csv_file(self):
        """Setup CSV file with headers"""
        os.makedirs('logs', exist_ok=True)
        headers = ['id', 'green', 'brown', 'red']
        
        try:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            print(f"Error setting up CSV file: {str(e)}")
            raise
    
    def select_lines(self, frame):
        """Allow user to select the lines for speed detection"""
        print("\nSelect lines for vehicle speed detection in this order:")
        print("1. Blue Line (reference line)")
        print("2. Green Line")
        print("3. Brown Line")
        print("4. Red Line")
        print("Click to select start and end points for each line.")
        
        # Create a copy of the frame for drawing
        frame_copy = frame.copy()
        
        # Initialize points list
        points = []
        
        # Create window
        cv2.namedWindow('Select Lines', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Lines', 1280, 720)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 8:  # 2 points per line, 4 lines
                    points.append((x, y))
                    
                    # Create a fresh copy for display
                    display_frame = frame_copy.copy()
                    
                    # Draw all points and lines
                    for i, point in enumerate(points):
                        # Draw point
                        cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                        
                        # Add label
                        line_idx = i // 2
                        point_idx = i % 2
                        line_names = ['Blue', 'Green', 'Brown', 'Red']
                        point_names = ['Start', 'End']
                        label = f"{line_names[line_idx]} {point_names[point_idx]}"
                        cv2.putText(display_frame, label, 
                                  (point[0]+10, point[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw completed lines
                    for i in range(0, len(points), 2):
                        if i + 1 < len(points):
                            line_idx = i // 2
                            line_colors = [(255, 0, 0), (0, 255, 0), (19, 69, 139), (0, 0, 255)]
                            cv2.line(display_frame, points[i], points[i+1], line_colors[line_idx], 2)
                    
                    cv2.imshow('Select Lines', display_frame)
        
        cv2.setMouseCallback('Select Lines', mouse_callback)
        cv2.imshow('Select Lines', frame_copy)
        
        print("\nSelect 8 points in this order:")
        print("1. Blue Line Start")
        print("2. Blue Line End")
        print("3. Green Line Start")
        print("4. Green Line End")
        print("5. Brown Line Start")
        print("6. Brown Line End")
        print("7. Red Line Start")
        print("8. Red Line End")
        
        while len(points) < 8:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Store line coordinates
        if len(points) == 8:
            self.blue_line_start = points[0]
            self.blue_line_end = points[1]
            self.green_line_start = points[2]
            self.green_line_end = points[3]
            self.brown_line_start = points[4]
            self.brown_line_end = points[5]
            self.red_line_start = points[6]
            self.red_line_end = points[7]
            
            print("Lines selected successfully.")
            return True
        else:
            print("Not enough points selected. Using default line coordinates.")
            # Set default coordinates
            self.blue_line_start = (134, 267)
            self.blue_line_end = (805, 242)
            self.green_line_start = (112, 343)
            self.green_line_end = (966, 316)
            self.brown_line_start = (89, 419)
            self.brown_line_end = (1106, 374)
            self.red_line_start = (61, 507)
            self.red_line_end = (1268, 465)
            return False
    
    def calculate_speed(self, elapsed_time, distance):
        """Calculate speed in km/h from elapsed time and distance"""
        if elapsed_time > 0:
            speed_ms = distance / elapsed_time
            return speed_ms * 3.6  # Convert to km/h
        return 0
    
    def update_csv_data(self, vehicle_id):
        """Update CSV data for a vehicle"""
        if vehicle_id not in self.csv_data['id']:
            self.csv_data['id'].append(vehicle_id)
            self.csv_data['green'].append(self.vehicle_speeds[vehicle_id]['green'])
            self.csv_data['brown'].append(self.vehicle_speeds[vehicle_id]['brown'])
            self.csv_data['red'].append(self.vehicle_speeds[vehicle_id]['red'])
            print(f"Added/Updated vehicle {vehicle_id} to CSV with speeds: "
                  f"Green={self.vehicle_speeds[vehicle_id]['green']}, "
                  f"Brown={self.vehicle_speeds[vehicle_id]['brown']}, "
                  f"Red={self.vehicle_speeds[vehicle_id]['red']}")
        else:
            # Update existing entry
            idx = self.csv_data['id'].index(vehicle_id)
            self.csv_data['green'][idx] = self.vehicle_speeds[vehicle_id]['green']
            self.csv_data['brown'][idx] = self.vehicle_speeds[vehicle_id]['brown']
            self.csv_data['red'][idx] = self.vehicle_speeds[vehicle_id]['red']

    def process_frame(self, frame, vehicle_tracks):
        """Process a frame with tracked vehicles and return annotated frame"""
        try:
            processed_frame = frame.copy()
            current_time = time.time()
            
            # Draw all vehicles with consistent IDs
            for track in vehicle_tracks:
                x3, y3, x4, y4, id = track
                # Ensure coordinates are integers
                x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
                
                # Draw bounding box for all vehicles
                cv2.rectangle(processed_frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"ID:{id}", (x3, y3-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate center point
                cx = int((x3 + x4) // 2)
                cy = int((y3 + y4) // 2)
                cv2.circle(processed_frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # Initialize tracking for this vehicle if not already tracked
                if id not in self.vehicle_tracking:
                    self.vehicle_tracking[id] = {
                        'green': {'time': None, 'crossed': False},
                        'brown': {'time': None, 'crossed': False},
                        'red': {'time': None, 'crossed': False}
                    }
                    self.vehicle_speeds[id] = {'green': None, 'brown': None, 'red': None}
                
                # Check each line for crossing
                for line_name, line_start, line_end, distance in [
                    ('green', self.green_line_start, self.green_line_end, self.distance_green_blue),
                    ('brown', self.brown_line_start, self.brown_line_end, self.distance_brown_blue),
                    ('red', self.red_line_start, self.red_line_end, self.distance_red_blue)
                ]:
                    # Check if vehicle is near the line
                    if is_near_line(cx, cy, line_start, line_end, self.offset):
                        line_data = self.vehicle_tracking[id][line_name]
                        
                        # If this is the first time crossing this line
                        if not line_data['crossed']:
                            line_data['time'] = current_time
                            line_data['crossed'] = True
                            print(f"Vehicle {id} crossed {line_name} line")
                        elif line_data['time'] is not None:
                            # Calculate elapsed time since crossing
                            elapsed_time = current_time - line_data['time']
                            # Calculate and store speed if enough time has passed
                            if elapsed_time >= 0.1:  # Minimum time threshold
                                speed = self.calculate_speed(elapsed_time, distance)
                                self.vehicle_speeds[id][line_name] = int(speed)
                                print(f"Vehicle {id} speed at {line_name} line: {speed} km/h")
                                # Update CSV data immediately when we get a new speed
                                self.update_csv_data(id)
                
                # Draw speed information for all recorded speeds
                y_offset = y4 + 20
                for line_name in ['green', 'brown', 'red']:
                    if self.vehicle_speeds[id][line_name] is not None:
                        cv2.putText(processed_frame, 
                                  f'{line_name.capitalize()}: {self.vehicle_speeds[id][line_name]}Km/h', 
                                  (x4, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        y_offset += 20

    # Draw the lines
            if self.blue_line_start and self.blue_line_end:
                cv2.line(processed_frame, self.blue_line_start, self.blue_line_end, (255, 0, 0), 2)
                cv2.putText(processed_frame, 'Blue Line', self.blue_line_start, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            if self.green_line_start and self.green_line_end:
                cv2.line(processed_frame, self.green_line_start, self.green_line_end, (0, 255, 0), 2)
                cv2.putText(processed_frame, 'Green Line', self.green_line_start, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            if self.brown_line_start and self.brown_line_end:
                cv2.line(processed_frame, self.brown_line_start, self.brown_line_end, (19, 69, 139), 2)
                cv2.putText(processed_frame, 'Brown Line', self.brown_line_start, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            if self.red_line_start and self.red_line_end:
                cv2.line(processed_frame, self.red_line_start, self.red_line_end, (0, 0, 255), 2)
                cv2.putText(processed_frame, 'Red Line', self.red_line_start, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Draw counters
            cv2.putText(processed_frame, f'Vehicles Tracked: {len(self.vehicle_tracking)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            return processed_frame
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame
    
    def save_results_to_csv(self):
        """Save results to CSV file"""
        try:
# Save the speeds to CSV
            if self.csv_data['id']:
                df = pd.DataFrame(self.csv_data)
                df.to_csv(self.csv_filename, index=False)
                print(f"\nVehicle speed results saved to: {self.csv_filename}")
            else:
                print("No vehicle speed data to save in CSV.")
                
        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")
    
    def print_statistics(self):
        """Print final statistics"""
        total_vehicles = len(self.csv_data['id'])
        
        print("\nVehicle Speed Detection Statistics")
        print("================================")
        print(f"Total Unique Vehicles: {total_vehicles}")
        
        if total_vehicles > 0:
            print("\nSpeed Statistics:")
            for line in ['green', 'brown', 'red']:
                speeds = [s for s in self.csv_data[line] if s is not None]
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    max_speed = max(speeds)
                    print(f"{line.capitalize()} Line: Avg={avg_speed:.1f} km/h, Max={max_speed:.1f} km/h")
