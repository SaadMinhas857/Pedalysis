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
                
            # Calculate speeds for tracked vehicles
            current_tracks = {}
            speeds = {}
            
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                current_tracks[track_id] = center
                
                # Calculate speed if we have previous position
                if track_id in self.prev_tracks and self.pixels_per_meter is not None:
                    prev_center = self.prev_tracks[track_id]
                    distance_pixels = np.sqrt(
                        (center[0] - prev_center[0])**2 + 
                        (center[1] - prev_center[1])**2
                    )
                    distance_meters = distance_pixels / self.pixels_per_meter
                    speed = distance_meters * self.fps * 3.6  # Convert to km/h
                    speeds[track_id] = speed
                    
                    # Apply moving average to smooth speed estimates
                    if track_id in self.speed_estimates:
                        self.speed_estimates[track_id] = (
                            0.7 * self.speed_estimates[track_id] + 0.3 * speed
                        )
                    else:
                        self.speed_estimates[track_id] = speed
            
            # Update previous tracks
            self.prev_tracks = current_tracks
            
            # Draw detections and speeds on frame
            annotated_frame = self._draw_tracks(frame.copy(), tracks, speeds)
            
            return annotated_frame, {
                'tracks': tracks.tolist() if len(tracks) > 0 else [],
                'speeds': speeds
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
            
            cv2.putText(frame, speed_text, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
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
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Check if video file exists
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Video file not found: {args.source}")
    
    # Initialize detector
    detector = VehicleSpeedDetector(args.model, args.conf_thres)
    
    # Set camera parameters
    fps = int(30)  # You might want to get this from the video file
    detector.set_camera_parameters(args.pixels_per_meter, fps)
    
    # Open video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {args.source}")
    
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



class VehicleSpeedDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the Vehi