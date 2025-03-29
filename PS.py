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
    def __init__(self, pixels_per_meter: float):
        """Initialize speed detector"""
        self.pixels_per_meter = pixels_per_meter
        
        # Store pedestrian tracking data
        # {track_id: {'positions': [], 'speeds': [], 'times': []}}
        self.pedestrian_tracks = {}
    
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

    def process_track(self, track_id: int, position: Tuple[float, float], frame_time: float) -> Dict:
        """Process a track and return speed information"""
        # Initialize track data if new
        if track_id not in self.pedestrian_tracks:
            self.pedestrian_tracks[track_id] = {
                'positions': [],
                'speeds': [],
                'times': []
            }
        
        # Update track data
        track_data = self.pedestrian_tracks[track_id]
        track_data['positions'].append(position)
        track_data['times'].append(frame_time)
        
        # Calculate current speed
        current_speed = self.calculate_speed(
            track_data['positions'],
            track_data['times']
        )
        track_data['speeds'].append(current_speed)
        
        return {
            'current_speed': current_speed,
            'speeds': track_data['speeds']
        }

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
    detector = PedestrianSpeedDetector(args.pixels_per_meter)
    
    # Open video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {args.source}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
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