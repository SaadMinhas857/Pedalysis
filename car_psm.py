import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from tracker import Sort

class CarPSMDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize tracker
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Store lane polygons and target y-coordinate
        self.lane_polygons = []
        self.target_y = None
        
        # Store detections at target y-coordinate
        self.car_detections = {}  # {track_id: (lane_number, timestamp)}
        self.fps = None
    
    def select_lane_polygons(self, frame):
        """Allow user to select three lane polygons"""
        polygons = []
        current_polygon = []
        frame_copy = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                current_polygon.append((x, y))
                # Draw point
                cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)
                if len(current_polygon) > 1:
                    # Draw line from previous point
                    cv2.line(frame_copy, current_polygon[-2], current_polygon[-1], (0, 255, 0), 2)
                cv2.imshow('Select Lane Polygons', frame_copy)

        cv2.namedWindow('Select Lane Polygons')
        cv2.setMouseCallback('Select Lane Polygons', mouse_callback)
        
        print("Select points for three lane polygons. Press 'n' for next polygon, 'Enter' when done.")
        while len(polygons) < 3:
            cv2.imshow('Select Lane Polygons', frame_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n') and len(current_polygon) >= 3:
                # Complete current polygon
                cv2.line(frame_copy, current_polygon[-1], current_polygon[0], (0, 255, 0), 2)
                polygons.append(np.array(current_polygon))
                current_polygon = []
                print(f"Lane {len(polygons)} polygon completed. Select points for next lane.")
        
        cv2.destroyWindow('Select Lane Polygons')
        self.lane_polygons = polygons
        return polygons
    
    def select_target_y(self, frame):
        """Allow user to select target y-coordinate"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.target_y = y
                frame_copy = frame.copy()
                cv2.line(frame_copy, (0, y), (frame.shape[1], y), (0, 0, 255), 2)
                cv2.imshow('Select Target Y-Coordinate', frame_copy)
                cv2.waitKey(1000)
                cv2.destroyWindow('Select Target Y-Coordinate')

        cv2.namedWindow('Select Target Y-Coordinate')
        cv2.setMouseCallback('Select Target Y-Coordinate', mouse_callback)
        
        print("Click to select target y-coordinate")
        cv2.imshow('Select Target Y-Coordinate', frame)
        while self.target_y is None:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
    
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
    
    def get_lane_number(self, point):
        """Determine which lane a point is in"""
        for i, polygon in enumerate(self.lane_polygons):
            if self.point_in_polygon(point, polygon):
                return i + 1
        return None
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict]:
        try:
            # Run YOLOv8 inference
            results = self.model(frame)[0]
            
            # Process detections
            detections = []
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.conf_threshold and int(cls) == 4:  # Class 4 cars only
                    detections.append([x1, y1, x2, y2, conf])
            
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
                
                # Check if car crosses target y-coordinate
                if (track_id not in self.car_detections and 
                    abs(center_y - self.target_y) < 5):  # 5-pixel threshold
                    lane_num = self.get_lane_number((center_x, center_y))
                    if lane_num is not None:
                        self.car_detections[track_id] = (lane_num, current_time)
                        print(f"Car {track_id} detected in lane {lane_num} at time {current_time:.2f}s")
            
            # Draw visualization
            annotated_frame = self._draw_visualization(frame.copy(), tracks)
            
            return annotated_frame, {
                'tracks': tracks.tolist() if len(tracks) > 0 else [],
                'detections': self.car_detections
            }
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            return frame, {'tracks': [], 'detections': {}}
    
    def _draw_visualization(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        # First draw the lane polygons with transparency
        overlay = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Different color for each lane
        for i, polygon in enumerate(self.lane_polygons):
            cv2.fillPoly(overlay, [polygon], colors[i])
            # Add lane number
            center = np.mean(polygon, axis=0).astype(int)
            cv2.putText(frame, f"Lane {i+1}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Blend the overlay with original frame
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw target y-coordinate line with dashed effect
        for x in range(0, frame.shape[1], 20):  # Draw dashed line
            cv2.line(frame, (x, self.target_y), (x + 10, self.target_y),
                    (0, 0, 255), 2)
        
        # Draw tracked vehicles
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Draw ID and detection info
            text_lines = []
            text_lines.append(f"ID: {track_id}")
            
            if track_id in self.car_detections:
                lane_num, timestamp = self.car_detections[track_id]
                text_lines.append(f"Lane: {lane_num}")
                text_lines.append(f"Time: {timestamp:.2f}s")
            
            # Draw text with background
            text_y = int(y1) - 10
            for text in text_lines:
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw background rectangle
                cv2.rectangle(frame, 
                            (int(x1), text_y - text_height - 5),
                            (int(x1) + text_width, text_y + 5),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, text,
                           (int(x1), text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                text_y -= text_height + 10
            
            # Draw center point of vehicle
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
        
        # Add frame information at the top
        cv2.putText(frame,
                    f"Target Y-coordinate: {self.target_y}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add detection count
        detected_count = len([k for k, v in self.car_detections.items() 
                             if v[1] is not None])
        cv2.putText(frame,
                    f"Detected Cars: {detected_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description='Car PSM Detection using YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to input video file')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output_psm.mp4', help='Path to output video')
    parser.add_argument('--display', action='store_true', help='Display output in real-time')
    args = parser.parse_args()
    
    # Initialize detector
    detector = CarPSMDetector(args.model, args.conf_thres)
    
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
    
    # Select lane polygons and target y-coordinate
    detector.select_lane_polygons(first_frame)
    detector.select_target_y(first_frame)
    
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
                    # Add frame counter and tracking info
                    cv2.putText(
                        processed_frame,
                        f"Frame: {frame_count} | FPS: {fps} | Tracked Cars: {len(results['tracks'])}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.imshow('Car PSM Detection', processed_frame)
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