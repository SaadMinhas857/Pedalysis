import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from tracker import Sort
import csv
from datetime import datetime
import os

class PedestrianPSMDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize tracker
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Store lane polygons
        self.lane_polygons = []
        
        # Store pedestrian tracking data
        # {track_id: {lane_num: {'entry': time, 'middle': time, 'exit': time, 'direction': str}}}
        self.pedestrian_tracks = {}
        self.fps = None
        
        # Store current lane for each pedestrian
        self.current_lanes = {}  # {track_id: current_lane}
        
        # Add CSV logging setup
        self.csv_file = None
        self.csv_writer = None
        self.written_tracks = set()  # Add this to keep track of which IDs we've written
        self.direction_history = {}  # Add this line
        self.setup_csv_logging()
    
    def setup_csv_logging(self):
        """Setup CSV file with track-based format including direction"""
        os.makedirs('logs', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = open(f'logs/pedestrian_crossings_{timestamp}.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Updated headers with direction
        headers = ['Track_ID', 'Direction']
        for lane in range(3):  # 3 lanes
            headers.extend([
                f'Lane{lane+1}_Entry',
                f'Lane{lane+1}_Middle',
                f'Lane{lane+1}_Exit'
            ])
        self.csv_writer.writerow(headers)
    
    def write_tracks_to_csv(self):
        """Write completed tracks to CSV with direction information"""
        for track_id, lanes in self.pedestrian_tracks.items():
            if track_id in self.written_tracks:
                continue
                
            # Check if any lane has been completed
            is_complete = False
            direction = None
            for lane_data in lanes.values():
                if all(lane_data.get(pos) is not None for pos in ['entry', 'middle', 'exit']):
                    is_complete = True
                    direction = lane_data.get('direction')
                    break
            
            if is_complete and direction:
                row = [track_id, direction]  # Start with track ID and direction
                
                # Add times for each lane
                for lane_num in range(3):  # 3 lanes
                    if lane_num in lanes:
                        times = lanes[lane_num]
                        row.extend([
                            f"{times.get('entry', ''):.2f}" if times.get('entry') is not None else "",
                            f"{times.get('middle', ''):.2f}" if times.get('middle') is not None else "",
                            f"{times.get('exit', ''):.2f}" if times.get('exit') is not None else ""
                        ])
                    else:
                        row.extend(["", "", ""])  # Empty values for lanes not crossed
                
                self.csv_writer.writerow(row)
                self.csv_file.flush()
                self.written_tracks.add(track_id)
    
    def select_lane_polygons(self, frame):
        """Allow user to select three lane polygons with exactly 4 points each"""
        polygons = []
        current_polygon = []
        frame_copy = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(current_polygon) < 4:
                    current_polygon.append((x, y))
                    # Draw point with number
                    point_num = len(current_polygon)
                    cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(frame_copy, str(point_num), (x+5, y+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if len(current_polygon) > 1:
                        # Draw line from previous point
                        cv2.line(frame_copy, current_polygon[-2], current_polygon[-1], (0, 255, 0), 2)
                    if len(current_polygon) == 4:
                        # Complete polygon
                        cv2.line(frame_copy, current_polygon[-1], current_polygon[0], (0, 255, 0), 2)
                    cv2.imshow('Select Lane Polygons', frame_copy)

        cv2.namedWindow('Select Lane Polygons')
        cv2.setMouseCallback('Select Lane Polygons', mouse_callback)
        
        print("\nInstructions for selecting lane polygons:")
        print("For each lane, select 4 corners in this order:")
        print("1. Bottom-right corner")
        print("2. Bottom-left corner")
        print("3. Top-left corner")
        print("4. Top-right corner")
        print("\nNote: Points should be selected counter-clockwise to properly detect:")
        print("- Left side: Entry zone")
        print("- Middle: Middle zone")
        print("- Right side: Exit zone")
        print("\nPress 'n' after selecting all 4 points to move to the next lane.")
        
        while len(polygons) < 3:
            cv2.imshow('Select Lane Polygons', frame_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n') and len(current_polygon) == 4:
                polygons.append(np.array(current_polygon))
                current_polygon = []
                print(f"Lane {len(polygons)} polygon completed. Select points for next lane.")
                # Create new copy of frame for next polygon
                frame_copy = frame.copy()
                # Draw previous polygons
                for poly in polygons:
                    cv2.polylines(frame_copy, [poly], True, (0, 255, 0), 2)
        
        cv2.destroyWindow('Select Lane Polygons')
        
        # After selecting all polygons, show a preview
        preview_frame = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        for i, polygon in enumerate(polygons):
            # Draw filled polygon with transparency
            overlay = preview_frame.copy()
            cv2.fillPoly(overlay, [polygon], colors[i])
            preview_frame = cv2.addWeighted(overlay, 0.3, preview_frame, 0.7, 0)
            
            # Draw polygon outline
            cv2.polylines(preview_frame, [polygon], True, colors[i], 2)
            
            # Add lane number
            center = np.mean(polygon, axis=0).astype(int)
            cv2.putText(preview_frame, f"Lane {i+1}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow('Lane Preview', preview_frame)
        print("Preview of lane polygons. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        self.lane_polygons = polygons
        return polygons
    
    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            if ((polygon[i][1] > y) != (polygon[j][1] > y) and
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                 (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
                inside = not inside
            j = i
            
        return inside
    
    def get_lane_position(self, point, lane_num):
        """Determine position within lane with smaller zones and empty spaces"""
        polygon = self.lane_polygons[lane_num]
        x, _ = point
        
        # Get x-coordinates of polygon
        x_coords = [p[0] for p in polygon]
        min_x = min(x_coords)
        max_x = max(x_coords)
        total_width = max_x - min_x
        
        # Define zone widths
        entry_width = total_width * 0.05  # 5% for entry
        middle_width = total_width * 0.10  # 10% for middle
        exit_width = total_width * 0.05   # 5% for exit
        
        # Calculate zone positions
        entry_zone_end = min_x + entry_width
        middle_zone_start = min_x + (total_width - middle_width) / 2
        middle_zone_end = middle_zone_start + middle_width
        exit_zone_start = max_x - exit_width
        
        # Determine position
        if min_x <= x <= entry_zone_end:
            return 'entry'
        elif middle_zone_start <= x <= middle_zone_end:
            return 'middle'
        elif exit_zone_start <= x <= max_x:
            return 'exit'
        else:
            return None  # In empty space
    
    def get_lane_number(self, point):
        """Determine which lane a point is in"""
        for i, polygon in enumerate(self.lane_polygons):
            if self.point_in_polygon(point, polygon):
                print(f"Point {point} is in Lane {i+1}")  # Debug print
                return i
        print(f"Point {point} is not in any lane")  # Debug print
        return None
    
    def determine_direction(self, track_id, current_x):
        """Determine movement direction based on x-coordinate history"""
        if track_id not in self.direction_history:
            self.direction_history[track_id] = []
        
        self.direction_history[track_id].append(current_x)
        
        if len(self.direction_history[track_id]) >= 10:  # Use last 10 points
            x_coords = self.direction_history[track_id][-10:]
            if x_coords[-1] - x_coords[0] > 10:  # Moving right
                return 'left_to_right'
            elif x_coords[0] - x_coords[-1] > 10:  # Moving left
                return 'right_to_left'
        return None

    def _draw_enhanced_visualization(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """Enhanced visualization with more detailed track data and lane information"""
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw lane polygons with transparency
        overlay = vis_frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Colors for each lane
        
        # Draw lanes with labels for entry, middle, exit zones
        for i, polygon in enumerate(self.lane_polygons):
            # Draw filled polygon with transparency
            cv2.fillPoly(overlay, [polygon], colors[i])
            
            # Get lane boundaries for zone visualization
            x_coords = [p[0] for p in polygon]
            min_x = min(x_coords)
            max_x = max(x_coords)
            total_width = max_x - min_x
            
            # Calculate zone positions
            entry_width = total_width * 0.05
            middle_width = total_width * 0.10
            exit_width = total_width * 0.05
            
            entry_zone_end = min_x + entry_width
            middle_zone_start = min_x + (total_width - middle_width) / 2
            middle_zone_end = middle_zone_start + middle_width
            exit_zone_start = max_x - exit_width
            
            # Draw zone markers
            bottom_y = max([p[1] for p in polygon])
            top_y = min([p[1] for p in polygon])
            
            # Draw zone areas with different colors
            zone_overlay = vis_frame.copy()
            
            # Entry zone (green)
            entry_pts = np.array([
                [min_x, bottom_y],
                [entry_zone_end, bottom_y],
                [entry_zone_end, top_y],
                [min_x, top_y]
            ], np.int32)
            cv2.fillPoly(zone_overlay, [entry_pts], (0, 255, 0))
            
            # Middle zone (blue)
            middle_pts = np.array([
                [middle_zone_start, bottom_y],
                [middle_zone_end, bottom_y],
                [middle_zone_end, top_y],
                [middle_zone_start, top_y]
            ], np.int32)
            cv2.fillPoly(zone_overlay, [middle_pts], (255, 0, 0))
            
            # Exit zone (red)
            exit_pts = np.array([
                [exit_zone_start, bottom_y],
                [max_x, bottom_y],
                [max_x, top_y],
                [exit_zone_start, top_y]
            ], np.int32)
            cv2.fillPoly(zone_overlay, [exit_pts], (0, 0, 255))
            
            # Blend zone overlay
            vis_frame = cv2.addWeighted(zone_overlay, 0.3, vis_frame, 0.7, 0)
            
            # Add zone labels
            cv2.putText(vis_frame, "Entry", (int(min_x), bottom_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_frame, "Middle", (int(middle_zone_start), bottom_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_frame, "Exit", (int(exit_zone_start), bottom_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend the overlay with original frame
        alpha = 0.3
        vis_frame = cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0)
        
        # Add timestamp and tracking stats
        cv2.putText(vis_frame, f"Pedestrians: {len(tracks)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"Total Tracked: {len(self.pedestrian_tracks)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw tracked pedestrians with enhanced information
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Get center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Determine direction status
            direction = None
            if track_id in self.direction_history and len(self.direction_history[track_id]) >= 2:
                if self.direction_history[track_id][-1] > self.direction_history[track_id][0]:
                    direction = "→"  # right
                else:
                    direction = "←"  # left
            
            # Choose bounding box color based on direction
            box_color = (0, 255, 0)  # Default green
            if track_id in self.pedestrian_tracks:
                for lane_data in self.pedestrian_tracks[track_id].values():
                    if lane_data.get('direction') == 'left_to_right':
                        box_color = (0, 255, 255)  # Yellow for left to right
                    elif lane_data.get('direction') == 'right_to_left':
                        box_color = (255, 0, 255)  # Purple for right to left
            
            # Draw bounding box with thicker lines
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            
            # Draw track ID with background
            id_bg = (0, 0, 0)
            cv2.rectangle(vis_frame, (int(x1), int(y1)-25), (int(x1)+40, int(y1)), id_bg, -1)
            cv2.putText(vis_frame, f"ID:{track_id}", (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Draw direction arrow if known
            if direction:
                cv2.putText(vis_frame, direction, (int(x1)+45, int(y1)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
            # Draw center point
            cv2.circle(vis_frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # Draw trail of recent positions
            if track_id in self.direction_history and len(self.direction_history[track_id]) > 1:
                # Get last 20 positions or all if less
                history_length = min(20, len(self.direction_history[track_id]))
                positions = []
                
                # Calculate y positions based on stored x positions
                for x_pos in self.direction_history[track_id][-history_length:]:
                    # Use current y since we only store x in history
                    positions.append((int(x_pos), center_y))
                
                # Draw connected trail
                for i in range(1, len(positions)):
                    # Increasing intensity as we get closer to current position
                    intensity = int(255 * i / len(positions))
                    cv2.line(vis_frame, positions[i-1], positions[i], (0, intensity, intensity), 2)
            
            # Draw detailed info box if pedestrian has crossed at least one lane
            if track_id in self.pedestrian_tracks and self.pedestrian_tracks[track_id]:
                # Create a list of text lines
                text_lines = []
                
                # Add lane crossing times
                for lane_num, times in self.pedestrian_tracks[track_id].items():
                    direction_text = ""
                    if times.get('direction'):
                        direction_text = f" ({times['direction']})"
                    
                    text_lines.append(f"Lane {lane_num+1}{direction_text}:")
                    
                    if times.get('entry'):
                        text_lines.append(f"  Entry: {times['entry']:.2f}s")
                    if times.get('middle'):
                        text_lines.append(f"  Mid:   {times['middle']:.2f}s")
                    if times.get('exit'):
                        text_lines.append(f"  Exit:  {times['exit']:.2f}s")
                
                # Calculate info box dimensions
                line_height = 20
                box_height = len(text_lines) * line_height + 10
                box_width = 180
                
                # Position box to the right of bounding box
                box_x = int(x2) + 5
                box_y = int(y1)
                
                # Ensure box stays within frame
                if box_x + box_width > vis_frame.shape[1]:
                    box_x = int(x1) - box_width - 5
                if box_y + box_height > vis_frame.shape[0]:
                    box_y = vis_frame.shape[0] - box_height - 5
                
                # Draw info box background
                cv2.rectangle(vis_frame, 
                            (box_x, box_y), 
                            (box_x + box_width, box_y + box_height), 
                            (0, 0, 0), -1)
                
                # Draw text lines
                for i, text in enumerate(text_lines):
                    y_pos = box_y + 20 + i * line_height
                    cv2.putText(vis_frame, text, (box_x + 5, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add legend at the bottom
        legend_y = vis_frame.shape[0] - 60
        cv2.putText(vis_frame, "Legend:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(vis_frame, (90, legend_y-15), (120, legend_y+5), (0, 255, 255), -1)
        cv2.putText(vis_frame, "Left to Right", (125, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(vis_frame, (250, legend_y-15), (280, legend_y+5), (255, 0, 255), -1)
        cv2.putText(vis_frame, "Right to Left", (285, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict]:
        try:
            # Run YOLOv8 inference
            results = self.model(frame)[0]
            
            # Process detections
            detections = []
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.conf_threshold and int(cls) == 3:  # Changed back to class 3 for pedestrians
                    detections.append([x1, y1, x2, y2, conf])
            
            print(f"Frame {frame_id}: Found {len(detections)} pedestrians")  # Debug print
            
            # Update tracker
            if len(detections) > 0:
                tracks = self.tracker.update(np.array(detections))
            else:
                tracks = np.empty((0, 5))
            
            # Process each track
            current_time = frame_id / self.fps
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                current_point = (center_x, center_y)
                
                # Debug print for track position
                print(f"Track {track_id} at position {current_point}")
                
                # Initialize track data if new
                if track_id not in self.pedestrian_tracks:
                    self.pedestrian_tracks[track_id] = {}
                    self.direction_history[track_id] = []
                
                # Determine direction
                direction = self.determine_direction(track_id, center_x)
                
                # Check current lane
                current_lane = self.get_lane_number(current_point)
                if current_lane is not None:
                    print(f"Track {track_id} detected in Lane {current_lane + 1}")  # Debug print
                    # Initialize lane data if new
                    if current_lane not in self.pedestrian_tracks[track_id]:
                        self.pedestrian_tracks[track_id][current_lane] = {
                            'entry': None,
                            'middle': None,
                            'exit': None,
                            'direction': None
                        }
                    
                    # Get position in lane based on direction
                    if direction == 'left_to_right':
                        position = self.get_lane_position(current_point, current_lane)
                    elif direction == 'right_to_left':
                        raw_position = self.get_lane_position(current_point, current_lane)
                        if raw_position is not None:  # Only map if in a zone
                            position = {'entry': 'exit', 'exit': 'entry', 'middle': 'middle'}[raw_position]
                        else:
                            position = None
                    else:
                        continue  # Skip if direction not yet determined
                    
                    # Only update timestamps if in a valid zone
                    if position is not None:
                        if self.pedestrian_tracks[track_id][current_lane][position] is None:
                            self.pedestrian_tracks[track_id][current_lane][position] = current_time
                            self.pedestrian_tracks[track_id][current_lane]['direction'] = direction
                            print(f"Pedestrian {track_id} ({direction}) {position} Lane {current_lane+1} at {current_time:.2f}s")
            
            # Draw visualization (replace this line)
            # annotated_frame = self._draw_visualization(frame.copy(), tracks)
            
            # With enhanced visualization
            annotated_frame = self._draw_enhanced_visualization(frame.copy(), tracks)
            
            # Add debug print for tracks
            print(f"Frame {frame_id}: Tracking {len(tracks)} pedestrians")
            
            return annotated_frame, {
                'tracks': tracks.tolist() if len(tracks) > 0 else [],
                'pedestrian_tracks': self.pedestrian_tracks
            }
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            return frame, {'tracks': [], 'pedestrian_tracks': {}}
    
    def __del__(self):
        """Cleanup method to ensure CSV file is properly closed"""
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()

def main():
    parser = argparse.ArgumentParser(description='Pedestrian PSM Detection using YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to input video file')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output_pedestrian_psm.mp4', help='Path to output video')
    parser.add_argument('--display', action='store_true', help='Display output in real-time')
    args = parser.parse_args()
    
    # Initialize detector
    detector = PedestrianPSMDetector(args.model, args.conf_thres)
    
    # Open video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {args.source}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    detector.fps = fps
    
    # Read first frame for setup
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    # Select lane polygons
    detector.select_lane_polygons(first_frame)
    
    # Reset video capture
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
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, results = detector.process_frame(frame, frame_count)
            
            # Write frame
            writer.write(processed_frame)
            
            # Display frame if requested
            if args.display:
                cv2.imshow('Pedestrian PSM Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write CSV less frequently (every 60 frames instead of 30)
            if frame_count % 60 == 0:
                detector.write_tracks_to_csv()
            
            frame_count += 1
            
    finally:
        # Write final CSV before closing
        detector.write_tracks_to_csv()
        if detector.csv_file:
            detector.csv_file.close()
        
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        # Print final crossing statistics and CSV file location
        print("\nPedestrian Crossing Statistics:")
        for ped_id, lanes in detector.pedestrian_tracks.items():
            print(f"\nPedestrian {ped_id}:")
            for lane_num, times in lanes.items():
                print(f"  Lane {lane_num + 1}:")
                for position, time in times.items():
                    if time is not None:
                        print(f"    {position}: {time:.2f}s")
        
        print(f"\nProcessed {frame_count} frames")
        print(f"Video output saved to: {output_path}")
        print(f"Crossing data saved to: logs/pedestrian_crossings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

if __name__ == "__main__":
    main() 