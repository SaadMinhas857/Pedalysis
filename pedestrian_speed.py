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

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict]:
        try:
            # Run YOLOv8 inference
            results = self.model(frame)[0]
            
            # Process detections
            detections = []
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.conf_threshold and int(cls) == 3:  # Class 2 for pedestrians
                    detections.append([x1, y1, x2, y2, conf])
            
            # Update tracker
            if len(detections) > 0:
                tracks = self.tracker.update(np.array(detections))
            else:
                tracks = np.empty((0, 5))
            
            # Create visualization frame
            vis_frame = frame.copy()
            current_time = frame_id / self.fps if self.fps else 0
            
            # Process each track
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Initialize track data if new
                if track_id not in self.pedestrian_tracks:
                    self.pedestrian_tracks[track_id] = {
                        'positions': [],
                        'speeds': [],
                        'times': []
                    }
                
                # Update track data
                track_data = self.pedestrian_tracks[track_id]
                track_data['positions'].append((center_x, center_y))
                track_data['times'].append(current_time)
                
                # Calculate speed
                current_speed = self.calculate_speed(
                    track_data['positions'],
                    track_data['times']
                )
                track_data['speeds'].append(current_speed)
                
                # Get color for this track
                color = self.colors[track_id % len(self.colors)]
                color = (int(color[0]), int(color[1]), int(color[2]))
                
                # Draw bounding box
                cv2.rectangle(vis_frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                
                # Draw trajectory
                positions = track_data['positions']
                for i in range(1, len(positions)):
                    pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                    pt2 = (int(positions[i][0]), int(positions[i][1]))
                    cv2.line(vis_frame, pt1, pt2, color, 2)
                
                # Draw ID and speed
                speed_text = f"ID:{track_id} {current_speed:.1f} m/s"
                cv2.putText(vis_frame, speed_text,
                           (int(bbox[0]), int(bbox[1]-10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw direction arrow
                if len(positions) >= 2:
                    dx = positions[-1][0] - positions[-2][0]
                    if abs(dx) > 1:  # Only draw if there's significant horizontal movement
                        arrow_length = 30
                        arrow_x = int(bbox[0] + bbox[2]) // 2
                        arrow_y = int(bbox[1] + bbox[3]) // 2
                        if dx > 0:  # Moving right
                            cv2.arrowedLine(vis_frame,
                                          (arrow_x - arrow_length, arrow_y),
                                          (arrow_x + arrow_length, arrow_y),
                                          color, 2, tipLength=0.3)
                        else:  # Moving left
                            cv2.arrowedLine(vis_frame,
                                          (arrow_x + arrow_length, arrow_y),
                                          (arrow_x - arrow_length, arrow_y),
                                          color, 2, tipLength=0.3)
            
            # Add frame info
            cv2.putText(vis_frame, f"Frame: {frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Tracked: {len(tracks)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return vis_frame, {
                'tracks': tracks.tolist() if len(tracks) > 0 else [],
                'pedestrian_tracks': self.pedestrian_tracks
            }
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            return frame, {'tracks': [], 'pedestrian_tracks': {}}

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
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    detector.fps = fps
    
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
                cv2.imshow('Pedestrian Speed Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
    finally:
        cap.release()
        writer.release()
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
        print(f"Video output saved to: {output_path}")

if __name__ == "__main__":
    main() 