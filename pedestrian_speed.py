import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from tracker import Sort
import math

class PedestrianSpeedDetector:
    def __init__(self, model_path: str, pixels_per_meter: float, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.pixels_per_meter = pixels_per_meter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize tracker
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        
        # Store pedestrian tracking data
        # {track_id: {'positions': [], 'speeds': [], 'times': []}}
        self.pedestrian_tracks = {}
        self.fps = None
        
        # Colors for visualization
        self.colors = np.random.randint(0, 255, size=(100, 3))
        
        # Add perspective transform matrix
        self.perspective_matrix = None
        self.transformed_size = (1080, 1080)
        self.transformed_frame = np.zeros((1080, 1080, 3), dtype=np.uint8)
    
    def calculate_speed(self, positions: List[Tuple[float, float]], times: List[float]) -> float:
        """Calculate speed in meters per second using last N positions"""
        if len(positions) < 2:
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
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        # Convert distance to meters
        distance_meters = total_distance / self.pixels_per_meter
        
        # Calculate time difference
        time_diff = recent_times[-1] - recent_times[0]
        
        if time_diff > 0:
            speed = distance_meters / time_diff
            return speed
        return 0.0

    def select_transform_points(self, frame):
        """Allow user to select four points for perspective transform"""
        points = []
        frame_copy = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                # Draw point
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame_copy, str(len(points)), (x+10, y+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if len(points) > 1:
                    # Draw lines connecting points
                    cv2.line(frame_copy, points[-2], points[-1], (0, 255, 0), 2)
                if len(points) == 4:
                    # Connect last point to first point
                    cv2.line(frame_copy, points[-1], points[0], (0, 255, 0), 2)
                cv2.imshow('Select Transform Points', frame_copy)

        cv2.namedWindow('Select Transform Points')
        cv2.setMouseCallback('Select Transform Points', mouse_callback)
        
        print("\nSelect 4 points for perspective transform in this order:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")
        
        while len(points) < 4:
            cv2.imshow('Select Transform Points', frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow('Select Transform Points')
        
        if len(points) == 4:
            # Define destination points for 1080x1080 frame
            dst_points = np.float32([
                [0, 0],  # Top-left
                [1080, 0],  # Top-right
                [1080, 1080],  # Bottom-right
                [0, 1080]  # Bottom-left
            ])
            
            # Calculate perspective transform matrix
            src_points = np.float32(points)
            self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            return True
        return False

    def transform_point(self, point):
        """Transform a single point using perspective matrix"""
        if self.perspective_matrix is None:
            return point
        
        transformed = cv2.perspectiveTransform(
            np.array([[point]], dtype=np.float32),
            self.perspective_matrix
        )
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    def transform_bbox(self, bbox):
        """Transform bounding box coordinates using perspective matrix"""
        if self.perspective_matrix is None:
            return bbox
        
        # Get bottom center of bounding box (feet position)
        foot_point = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
        
        # Transform foot point
        transformed_foot = self.transform_point(foot_point)
        
        # Create a small box around the transformed point
        box_size = 30  # Size of box in transformed view
        x1 = transformed_foot[0] - box_size//2
        y1 = transformed_foot[1] - box_size
        x2 = transformed_foot[0] + box_size//2
        y2 = transformed_foot[1]
        
        return [x1, y1, x2, y2]

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        try:
            # Run YOLOv8 inference on original frame
            results = self.model(frame)[0]
            
            # Create original visualization frame
            original_frame = frame.copy()
            
            # Transform the frame to bird's eye view
            transformed_frame = cv2.warpPerspective(
                frame, 
                self.perspective_matrix, 
                self.transformed_size
            )
            self.transformed_frame = transformed_frame
            
            # Process detections
            detections = []
            transformed_detections = []
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.conf_threshold and int(cls) == 2:
                    # Store original detection
                    detections.append([x1, y1, x2, y2, conf])
                    
                    # Transform detection for bird's eye view
                    transformed_bbox = self.transform_bbox([x1, y1, x2, y2])
                    transformed_detections.append(transformed_bbox + [conf])
            
            # Update trackers
            if len(detections) > 0:
                original_tracks = self.tracker.update(np.array(detections))
                transformed_tracks = self.tracker.update(np.array(transformed_detections))
            else:
                original_tracks = np.empty((0, 5))
                transformed_tracks = np.empty((0, 5))
            
            current_time = frame_id / self.fps if self.fps else 0
            
            # Process original view tracks
            for track in original_tracks:
                track_id = int(track[4])
                bbox = track[:4]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Initialize track data if new
                if track_id not in self.pedestrian_tracks:
                    self.pedestrian_tracks[track_id] = {
                        'positions': [],
                        'transformed_positions': [],
                        'speeds': [],
                        'transformed_speeds': [],
                        'times': []
                    }
                
                # Update track data for original view
                track_data = self.pedestrian_tracks[track_id]
                track_data['positions'].append((center_x, center_y))
                track_data['times'].append(current_time)
                
                # Calculate speed using original positions only
                current_speed = self.calculate_speed(
                    track_data['positions'],
                    track_data['times']
                )
                track_data['speeds'].append(current_speed)
                
                # Get color for this track
                color = self.colors[track_id % len(self.colors)]
                color = (int(color[0]), int(color[1]), int(color[2]))
                
                # Draw on original frame
                cv2.rectangle(original_frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                
                # Draw trajectory on original frame
                positions = track_data['positions']
                for i in range(1, len(positions)):
                    pt1 = positions[i-1]
                    pt2 = positions[i]
                    thickness = int(2 + (i / len(positions)) * 3)
                    cv2.line(original_frame, pt1, pt2, color, thickness)
                
                # Draw ID and original speed only in original view
                cv2.circle(original_frame, (center_x, center_y), 5, color, -1)
                speed_text = f"ID:{track_id} {current_speed:.1f} m/s"
                cv2.putText(original_frame, speed_text,
                           (int(bbox[0]), int(bbox[1]-10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Process transformed view tracks
            for track in transformed_tracks:
                track_id = int(track[4])
                bbox = track[:4]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int(bbox[3])  # Use bottom of box
                transformed_center = (center_x, center_y)
                
                # Get or initialize track data
                if track_id not in self.pedestrian_tracks:
                    self.pedestrian_tracks[track_id] = {
                        'positions': [],
                        'transformed_positions': [],
                        'speeds': [],
                        'transformed_speeds': [],
                        'times': []
                    }
                
                track_data = self.pedestrian_tracks[track_id]
                track_data['transformed_positions'].append(transformed_center)
                
                # Calculate speed using transformed positions only
                transformed_speed = self.calculate_speed(
                    track_data['transformed_positions'],
                    track_data['times']
                )
                track_data['transformed_speeds'].append(transformed_speed)
                
                # Get color for this track
                color = self.colors[track_id % len(self.colors)]
                color = (int(color[0]), int(color[1]), int(color[2]))
                
                # Draw bounding box on transformed frame
                cv2.rectangle(self.transformed_frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                
                # Draw trajectory on transformed frame
                positions = track_data['transformed_positions']
                for i in range(1, len(positions)):
                    pt1 = positions[i-1]
                    pt2 = positions[i]
                    thickness = int(2 + (i / len(positions)) * 3)
                    cv2.line(self.transformed_frame, pt1, pt2, color, thickness)
                
                # Draw ID and transformed speed only in transformed view
                cv2.circle(self.transformed_frame, transformed_center, 8, color, -1)
                speed_text = f"ID:{track_id} {transformed_speed:.1f} m/s"
                cv2.putText(self.transformed_frame, speed_text,
                           (transformed_center[0], transformed_center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add frame info to both views
            for frame_view, view_type, num_tracks in [
                (original_frame, "Original", len(original_tracks)),
                (self.transformed_frame, "Bird's Eye", len(transformed_tracks))
            ]:
                info_bg = frame_view.copy()
                cv2.rectangle(info_bg, (5, 5), (250, 70), (255, 255, 255), -1)
                cv2.addWeighted(info_bg, 0.6, frame_view, 0.4, 0, frame_view)
                
                cv2.putText(frame_view, f"{view_type} View - Frame: {frame_id}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame_view, f"Tracked: {num_tracks}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            return original_frame, self.transformed_frame, {
                'tracks': original_tracks.tolist() if len(original_tracks) > 0 else [],
                'pedestrian_tracks': self.pedestrian_tracks
            }
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            return frame, self.transformed_frame, {'tracks': [], 'pedestrian_tracks': {}}

def main():
    parser = argparse.ArgumentParser(description='Pedestrian Speed Detection using YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to input video file')
    parser.add_argument('--pixels-per-meter', type=float, required=True, help='Pixels per meter conversion factor')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output_speed.mp4', help='Path to output video')
    parser.add_argument('--display', action='store_true', help='Display output in real-time')
    args = parser.parse_args()
    
    # Initialize detector
    detector = PedestrianSpeedDetector(args.model, args.pixels_per_meter, args.conf_thres)
    
    # Open video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {args.source}")
    
    # Read first frame for perspective transform setup
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    # Select transform points
    if not detector.select_transform_points(first_frame):
        raise ValueError("Failed to select transform points")
    
    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    detector.fps = fps
    
    # Initialize video writer for both views
    original_output_path = Path('output_original.mp4')
    transformed_output_path = Path('output_transformed.mp4')
    
    original_writer = cv2.VideoWriter(
        str(original_output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    transformed_writer = cv2.VideoWriter(
        str(transformed_output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (1080, 1080)
    )
    
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            original_frame, transformed_frame, results = detector.process_frame(frame, frame_count)
            
            # Write frames
            original_writer.write(original_frame)
            transformed_writer.write(transformed_frame)
            
            # Display frames if requested
            if args.display:
                cv2.imshow('Original View', original_frame)
                cv2.imshow('Transformed View', transformed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
    finally:
        cap.release()
        original_writer.release()
        transformed_writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\nPedestrian Speed Statistics:")
        for ped_id, track_data in detector.pedestrian_tracks.items():
            speeds = track_data['speeds']
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
                print(f"\nPedestrian {ped_id}:")
                print(f"  Average Speed: {avg_speed:.2f} m/s")
                print(f"  Maximum Speed: {max_speed:.2f} m/s")
        
        print(f"\nProcessed {frame_count} frames")
        print(f"Original view saved to: {original_output_path}")
        print(f"Transformed view saved to: {transformed_output_path}")

if __name__ == "__main__":
    main() 