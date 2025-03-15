import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from tracker import Sort

class VehicleSpeedDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the Vehicle Speed Detector
        
        Args:
            model_path (str): Path to the YOLOv8 model weights
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Dictionary to store vehicle tracks
        self.vehicle_tracks = {}
        
        # Initialize tracker
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Store previous frame information for speed calculation
        self.prev_tracks = {}
        self.speed_estimates = {}
        
        # Camera parameters (to be calibrated)
        self.pixels_per_meter = None  # Will be set during calibration
        self.fps = None
        
        # Add these new variables for time tracking
        self.crossing_times = {}  # Dictionary to store crossing times for each vehicle
        self.speeds = {}  # Dictionary to store calculated speeds
        self.reference_lines = []
        self.real_distance = None
        
    def set_camera_parameters(self, pixels_per_meter: float, fps: float):
        """Set camera calibration parameters"""
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for vehicle detection and speed estimation
        
        Args:
            frame (np.ndarray): Input frame
            frame_id (int): Frame number
            
        Returns:
            Tuple[np.ndarray, Dict]: Processed frame with annotations and detection results
        """
        try:
            # Run YOLOv8 inference
            results = self.model(frame)[0]
            
            # Process detections
            detections = []
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.conf_threshold:
                    detections.append([x1, y1, x2, y2, conf])
            
            # Update tracker
            if len(detections) > 0:
                tracks = self.tracker.update(np.array(detections))
            else:
                tracks = np.empty((0, 5))
                
            # Calculate current time in seconds
            current_time = frame_id / self.fps
            
            # Process each track
            current_tracks = {}
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                current_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                current_tracks[track_id] = current_center
                
                # Check line crossings if we have previous position
                if track_id in self.prev_tracks:
                    prev_center = self.prev_tracks[track_id]
                    
                    # Initialize crossing times for new track
                    if track_id not in self.crossing_times:
                        self.crossing_times[track_id] = {'line1': None, 'line2': None}
                    
                    # Check first line crossing
                    if self.crossing_times[track_id]['line1'] is None:
                        if self._check_line_crossing(prev_center, current_center, self.reference_lines[0]):
                            self.crossing_times[track_id]['line1'] = current_time
                            print(f"Vehicle {track_id} crossed line 1 at time {current_time:.2f}s")
                    
                    # Check second line crossing
                    elif self.crossing_times[track_id]['line2'] is None:
                        if self._check_line_crossing(prev_center, current_center, self.reference_lines[1]):
                            self.crossing_times[track_id]['line2'] = current_time
                            # Calculate and store speed
                            time_diff = self.crossing_times[track_id]['line2'] - self.crossing_times[track_id]['line1']
                            if time_diff > 0:
                                speed = (self.real_distance / time_diff) * 3.6  # Convert to km/h
                                self.speeds[track_id] = speed
                                print(f"Vehicle {track_id} crossed line 2 at time {current_time:.2f}s")
                                print(f"Vehicle {track_id} speed: {speed:.2f} km/h")
            
            # Update previous tracks
            self.prev_tracks = current_tracks
            
            # Draw reference lines and tracks on frame
            frame = self._draw_reference_lines(frame.copy())
            annotated_frame = self._draw_tracks(frame, tracks, self.speeds)
            
            return annotated_frame, {
                'tracks': tracks.tolist() if len(tracks) > 0 else [],
                'speeds': self.speeds
            }
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            return frame, {'tracks': [], 'speeds': {}}
    
    def _draw_tracks(self, frame: np.ndarray, tracks: np.ndarray, speeds: Dict) -> np.ndarray:
        """Draw tracking boxes and speed information on the frame"""
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw ID and speed
            speed_text = f"ID: {track_id}"
            if track_id in speeds:
                speed_text += f" {speeds[track_id]:.1f} km/h"
            elif track_id in self.crossing_times:
                if self.crossing_times[track_id]['line1'] is not None:
                    speed_text += " (Crossed L1)"
            
            cv2.putText(frame, speed_text, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
        return frame

    def select_reference_lines(self, frame):
        """Allow user to select reference lines on the frame"""
        lines = []
        points = []
        frame_copy = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) % 2 == 0:  # Complete line
                    cv2.line(frame_copy, points[-2], points[-1], (0, 0, 255), 2)
                    lines.append((points[-2], points[-1]))
                    cv2.imshow('Select Reference Lines', frame_copy)
                else:  # First point of line
                    cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow('Select Reference Lines', frame_copy)

        cv2.namedWindow('Select Reference Lines')
        cv2.setMouseCallback('Select Reference Lines', mouse_callback)
        
        print("Select two reference lines (2 points each). Press 'Enter' when done.")
        while True:
            cv2.imshow('Select Reference Lines', frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(lines) == 2:  # Enter key and 2 lines selected
                break
        
        cv2.destroyWindow('Select Reference Lines')
        self.reference_lines = lines
        return lines

    def _check_line_crossing(self, prev_center, current_center, line):
        """Check if a track crossed the line"""
        line_p1, line_p2 = line
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Return true if line segments intersect
        return ccw(prev_center, line_p1, line_p2) != ccw(current_center, line_p1, line_p2) and \
               ccw(prev_center, current_center, line_p1) != ccw(prev_center, current_center, line_p2)

    def _draw_reference_lines(self, frame):
        """Draw reference lines on frame"""
        if len(self.reference_lines) >= 2:
            # Draw first line in red
            cv2.line(frame, 
                    self.reference_lines[0][0], 
                    self.reference_lines[0][1], 
                    (0, 0, 255), 2)
            # Draw second line in blue
            cv2.line(frame, 
                    self.reference_lines[1][0], 
                    self.reference_lines[1][1], 
                    (255, 0, 0), 2)
            
            # Add labels
            cv2.putText(frame, "Line 1", 
                       (self.reference_lines[0][0][0], self.reference_lines[0][0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Line 2", 
                       (self.reference_lines[1][0][0], self.reference_lines[1][0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return frame

def main():
    parser = argparse.ArgumentParser(description='Vehicle Speed Detection using YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to input video file')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video')
    parser.add_argument('--pixels-per-meter', type=float, required=True,
                       help='Pixels per meter ratio for speed calculation')
    parser.add_argument('--display', action='store_true', help='Display output in real-time')
    parser.add_argument('--distance', type=float, required=True,
                       help='Real-world distance between reference lines in meters')
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Check if video file exists
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Video file not found: {args.source}")
    
    # Initialize detector
    detector = VehicleSpeedDetector(args.model, args.conf_thres)
    
    # Open video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {args.source}")
    
    # Read first frame for reference line selection
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    # Select reference lines
    detector.select_reference_lines(first_frame)
    detector.real_distance = args.distance
    
    # Reset video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get actual FPS from video
    
    # Update detector FPS
    detector.set_camera_parameters(args.pixels_per_meter, fps)
    
    # Initialize video writer
    output_path = Path(args.output)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    frame_count = 0
    processed_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Process frame
                processed_frame, results = detector.process_frame(frame, frame_count)
                
                # Write frame
                writer.write(processed_frame)
                
                # Display frame if requested
                if args.display:
                    # Add frame counter and FPS to display
                    cv2.putText(
                        processed_frame,
                        f"Frame: {frame_count} | Tracked Objects: {len(results['tracks'])}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.imshow('Vehicle Speed Detection', processed_frame)
                    
                    # Break if 'q' is pressed or window is closed
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break
                
                processed_count += 1
                
                # Display progress
                if frame_count % 30 == 0:
                    print(f"Processed frame {frame_count}")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                writer.write(frame)  # Write original frame if processing fails
                
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {processed_count} frames out of {frame_count} total frames")
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main() 
