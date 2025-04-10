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
    def __init__(self, csv_filename: str):
        """Initialize Pedestrian Crossing Analyzer"""
        try:
            # Store points and lines
            self.lanes = []  # Will store 3 lanes, each with 4 points
            self.lane_lines = []  # Will store 9 vertical lines (3 per lane)
            self.crossing_times = {}  # {track_id: {0: time1, 1: time2, ..., 8: time9}}
            self.processed_ids = set()  # Keep track of fully processed IDs
            self.written_ids = set()  # Add this to track which IDs have been written to CSV
            
            # Add tracking for pedestrians and vehicles in Lane 3
            self.pedestrian_positions = {}  # {frame_number: [ped_ids]}
            self.vehicle_positions = {}     # {vehicle_id: {'positions': deque(maxlen=3), 'last_frame': frame_number}}
            self.stationary_vehicles = set()  # Set of vehicle IDs that are stationary
            self.parking_violations = set()   # Set of pedestrian IDs near stationary vehicles
            
            # Store CSV filename
            self.csv_filename = csv_filename
            
            # Setup CSV file
            self.setup_csv_file()
            
            print("Crossing Analyzer initialization complete.")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def setup_csv_file(self):
        """Setup CSV file with headers"""
        os.makedirs('logs', exist_ok=True)
        headers = ['Pedestrian_ID']
        # Add headers for all 9 line crossings (3 per lane)
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

    def select_points(self, frame):
        """Allow user to select points for three lanes"""
        points = []
        frame_copy = frame.copy()
        
        # Create and setup window properly
        cv2.namedWindow('Select Points', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Points', 1920, 1080)
        cv2.setWindowProperty('Select Points', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Define labels for all three lanes
        lane_labels = [
            ['Lane1 Top Left', 'Lane1 Top Right', 'Lane1 Bottom Right', 'Lane1 Bottom Left'],
            ['Lane2 Top Left', 'Lane2 Top Right', 'Lane2 Bottom Right', 'Lane2 Bottom Left'],
            ['Lane3 Top Left', 'Lane3 Top Right', 'Lane3 Bottom Right', 'Lane3 Bottom Left']
        ]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 12:
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
        
        print("\nSelect 12 points in this order:")
        for lane in range(3):
            for label in lane_labels[lane]:
                print(f"- {label}")
        
        while len(points) < 12:
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
        
        for lane_points in self.lanes:
            # Get top and bottom lines of the lane
            top_line = [lane_points[0], lane_points[1]]  # Left to right
            bottom_line = [lane_points[3], lane_points[2]]  # Left to right
            
            # Calculate three vertical lines (start, middle, end)
            for t in [0.2, 0.5, 0.8]:  # Changed from [0, 0.5, 1] to better distribute lines
                # Calculate top point
                top_x = int(top_line[0][0] + t * (top_line[1][0] - top_line[0][0]))
                top_y = int(top_line[0][1] + t * (top_line[1][1] - top_line[0][1]))
                
                # Calculate bottom point
                bottom_x = int(bottom_line[0][0] + t * (bottom_line[1][0] - bottom_line[0][0]))
                bottom_y = int(bottom_line[0][1] + t * (bottom_line[1][1] - bottom_line[0][1]))
                
                # Store line points
                self.lane_lines.append(((top_x, top_y), (bottom_x, bottom_y)))
                
                # Debug print for line creation
                lane_num = len(self.lane_lines) // 3
                line_num = len(self.lane_lines) % 3
                print(f"Created line {line_num + 1} for lane {lane_num + 1} at t={t}")
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
        """Check if point crosses any of the nine lines"""
        threshold = 7  # Increased threshold for better detection
        min_distance = float('inf')
        crossed_line = None
        
        # Check each line
        for line_num, line in enumerate(self.lane_lines):
            distance = self.point_to_line_distance(point, line[0], line[1])
            
            # Keep track of closest line
            if distance < min_distance:
                min_distance = distance
                if distance < threshold:
                    crossed_line = line_num
            
            # Debug print for all lines when point is close
            if distance < threshold + 5:
                lane = (line_num // 3) + 1
                line_in_lane = (line_num % 3) + 1
                print(f"Near Lane {lane} Line {line_in_lane}: distance={distance:.2f}, threshold={threshold}")
        
        if crossed_line is not None:
            lane = (crossed_line // 3) + 1
            line_in_lane = (crossed_line % 3) + 1
            print(f"Crossing detected at Lane {lane} Line {line_in_lane} (distance: {min_distance:.2f})")
            return crossed_line
            
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
            
            # Write any remaining results
            self.write_results()
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

    def write_track_to_csv(self, track_id):
        """Write track data to CSV"""
        if track_id in self.crossing_times:
            # Only write if pedestrian has crossed at least 2 lines
            if len(self.crossing_times[track_id]) < 2:
                return
                
            # Convert times to datetime objects for comparison
            try:
                times = [datetime.strptime(t, '%H:%M:%S') for t in self.crossing_times[track_id].values()]
                # Check if total crossing time is reasonable (between 1 second and 5 minutes)
                time_diff = (max(times) - min(times)).total_seconds()
                if not (1 <= time_diff <= 300):  # 1 second to 5 minutes
                    return
            except ValueError:
                return
            
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
                if row['Pedestrian_ID'] == str(track_id):
                    # Update line crossing times
                    for line_num, time in self.crossing_times[track_id].items():
                        lane = line_num // 3 + 1
                        line = line_num % 3 + 1
                        row[f'Lane{lane}_Line{line}_Time'] = time
                    updated = True
                    break
            
            if not updated:
                new_row = {'Pedestrian_ID': str(track_id)}
                # Initialize all columns to empty
                for header in headers:
                    new_row[header] = ''
                # Add line crossing times
                for line_num, time in self.crossing_times[track_id].items():
                    lane = line_num // 3 + 1
                    line = line_num % 3 + 1
                    new_row[f'Lane{lane}_Line{line}_Time'] = time
                rows.append(new_row)
            
            # Write back all data
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

    def write_results(self):
        """Write final results to CSV"""
        # Update any remaining unwritten tracks
        for track_id, crossings in self.crossing_times.items():
            if track_id not in self.csv_rows:
                times = [crossings.get(line, "") for line in range(9)]
                if any(times):  # Only include if at least one time exists
                    self.csv_rows[track_id] = times
        
        # Write final CSV
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Pedestrian_ID'] + [f'Lane{lane+1}_Line{line+1}_Time' for lane in range(3) for line in range(3)])
            for id, times in self.csv_rows.items():
                writer.writerow([id] + times)

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

    def draw_lines(self, frame):
        """Draw all crossing lines on frame"""
        for line in self.lane_lines:
            cv2.line(frame, line[0], line[1], (255, 0, 0), 2)

    def is_point_in_lane3(self, point):
        """Check if a point is inside Lane 3 polygon"""
        if len(self.lanes) < 3:
            return False
            
        lane3_points = self.lanes[2]  # Get Lane 3 points
        polygon = np.array(lane3_points, np.int32)
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def is_vehicle_stationary(self, positions):
        """Check if a vehicle is stationary based on its recent positions"""
        if len(positions) < 2:
            return False
            
        # Calculate maximum movement distance
        max_movement = 0
        positions_list = list(positions)
        for i in range(len(positions_list) - 1):
            dist = np.sqrt((positions_list[i+1][0] - positions_list[i][0])**2 + 
                         (positions_list[i+1][1] - positions_list[i][1])**2)
            max_movement = max(max_movement, dist)
            
        # If maximum movement is less than threshold (e.g., 5 pixels), consider it stationary
        return max_movement < 5

    def check_parking_violations(self, frame_number, pedestrians, vehicles):
        """Check for parking violations in Lane 3"""
        # Update pedestrian positions for this frame
        self.pedestrian_positions[frame_number] = []
        
        # Process pedestrians
        for ped in pedestrians:
            ped_id = int(ped[4])
            ped_center = (int((ped[0] + ped[2]) / 2), int((ped[1] + ped[3]) / 2))
            
            if self.is_point_in_lane3(ped_center):
                self.pedestrian_positions[frame_number].append(ped_id)
        
        # Process vehicles
        for vehicle in vehicles:
            vehicle_id = int(vehicle[4])
            vehicle_center = (int((vehicle[0] + vehicle[2]) / 2), int((vehicle[1] + vehicle[3]) / 2))
            
            if self.is_point_in_lane3(vehicle_center):
                if vehicle_id not in self.vehicle_positions:
                    self.vehicle_positions[vehicle_id] = {'positions': deque(maxlen=3), 'last_frame': frame_number}
                
                self.vehicle_positions[vehicle_id]['positions'].append(vehicle_center)
                self.vehicle_positions[vehicle_id]['last_frame'] = frame_number
                
                # Check if vehicle is stationary
                if self.is_vehicle_stationary(self.vehicle_positions[vehicle_id]['positions']):
                    self.stationary_vehicles.add(vehicle_id)
                else:
                    self.stationary_vehicles.discard(vehicle_id)
        
        # Clean up old vehicle entries
        current_vehicles = {int(v[4]) for v in vehicles}
        self.vehicle_positions = {k: v for k, v in self.vehicle_positions.items() 
                                if k in current_vehicles and frame_number - v['last_frame'] <= 3}
        
        # Check for consecutive frames with pedestrians
        if frame_number > 0 and frame_number - 1 in self.pedestrian_positions:
            prev_peds = self.pedestrian_positions[frame_number - 1]
            curr_peds = self.pedestrian_positions[frame_number]
            
            # If pedestrians are present in consecutive frames and there are stationary vehicles
            if prev_peds and curr_peds and self.stationary_vehicles:
                self.parking_violations.update(curr_peds)
        
        # Clean up old frame data
        self.pedestrian_positions = {k: v for k, v in self.pedestrian_positions.items() 
                                   if frame_number - k <= 2}

    def process_frame(self, frame, tracks, frame_time, vehicles=None):
        """Process a frame and return annotated frame"""
        # Create copy of frame for drawing
        processed_frame = frame.copy()
        
        # Draw all lanes and lines
        for lane_points in self.lanes:
            self.draw_lane_lines(processed_frame, lane_points)
        
        # Check for parking violations if vehicles data is provided
        if vehicles is not None:
            self.check_parking_violations(int(frame_time * 30), tracks, vehicles)  # Assuming 30 fps
        
        # Process tracks
        for track in tracks:
            track_id = int(track[4])
            bbox = track[:4]
            mid_x = int((bbox[0] + bbox[2]) / 2)
            mid_y = int((bbox[1] + bbox[3]) / 2)
            
            # Initialize crossing times for new track
            if track_id not in self.crossing_times:
                self.crossing_times[track_id] = {}
            
            # Check line crossings
            line_crossed = self.check_line_crossing((mid_x, mid_y), frame_time)
            if line_crossed is not None:
                # Record the time for this line crossing
                current_time = datetime.now().strftime('%H:%M:%S')
                
                # Only record if this line hasn't been crossed yet
                if line_crossed not in self.crossing_times[track_id]:
                    self.crossing_times[track_id][line_crossed] = current_time
                    lane = (line_crossed // 3) + 1
                    line_in_lane = (line_crossed % 3) + 1
                    print(f"Pedestrian {track_id} crossed line {line_in_lane} in lane {lane} at {current_time}")
                    
                    # Write to CSV after each new crossing
                    self.write_track_to_csv(track_id)
            
            # Draw visualization
            color = (0, 0, 255) if track_id in self.parking_violations else (0, 255, 0)
            cv2.rectangle(processed_frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), color, 2)
            
            # Add label for parking violation
            if track_id in self.parking_violations:
                cv2.putText(processed_frame, "PARKING", (int(bbox[0]), int(bbox[1])-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(processed_frame, f"ID:{track_id}", (int(bbox[0]), int(bbox[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(processed_frame, (mid_x, mid_y), 4, color, -1)
            
            # Add crossing information visualization
            if track_id in self.crossing_times:
                # Show total crossings
                crossing_count = len(self.crossing_times[track_id])
                cv2.putText(processed_frame, f"Crossings: {crossing_count}/9", 
                           (int(bbox[0]), int(bbox[3])+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return processed_frame

def main():
    try:
        # CHANGE THIS PATH to your video's location
        video_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\export6.mp4"
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        
        # CHANGE THIS PATH to your CSV file's location
        csv_filename = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\pedestrian_crossings_20230414_123456.csv"
        
        analyzer = PedestrianCrossingAnalyzer(csv_filename)
        analyzer.process_video(video_path)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return

if __name__ == "__main__":
    main() 