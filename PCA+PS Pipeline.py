import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from PCA import PedestrianCrossingAnalyzer
from PS import PedestrianSpeedDetector
from CARPSM import CarPSMDetector
from VCOMP import VehicleAnalyzer
from VS import VehicleSpeedDetector
import csv
from tracker import Sort

class PedestrianPipeline:
    def __init__(self, model_path: str, pixels_per_meter: float, conf_threshold: float = 0.7):
        """Initialize the main pipeline"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Store pixels_per_meter as instance variable
            self.pixels_per_meter = pixels_per_meter
            
            # Initialize YOLO model
            self.model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")
            
            # Initialize separate trackers for pedestrians and vehicles
            self.ped_tracker = Sort(
                max_age=60,       # Increased to maintain tracks longer when detection is lost
                min_hits=3,       # Reduced to start tracking sooner
                iou_threshold=0.3  # Reduced to be more lenient in matching
            )
            
            self.vehicle_tracker = Sort(
                max_age=60,
                min_hits=3,
                iou_threshold=0.3
            )
            
            # Initialize ID counters
            self.next_ped_id = 1
            self.next_vehicle_id = 1
            
            # Add appearance history for re-identification
            self.appearance_history = {}  # {track_id: [(bbox, frame_features)]}
            self.ped_id_mapping = {}  # {tracker_id: global_id}
            self.vehicle_id_mapping = {}  # {tracker_id: global_id}
            self.vehicle_classes = {}  # {global_id: class_id}
            self.temp_vehicle_classes = {}  # Temporary storage for frame detections
            
            # Setup CSV paths
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs('logs', exist_ok=True)
            
            # Create and store CSV paths with unique names
            self.ped_csv = f'logs/pedestrian_analysis_{timestamp}.csv'
            self.vehicle_csv = f'logs/vehicle_analysis_{timestamp}.csv'
            self.vcomp_csv = f'logs/vehicle_counts_{timestamp}.csv'
            self.vspeed_csv = f'logs/vehicle_speeds_{timestamp}.csv'
            
            print(f"CSV files will be saved to logs directory with timestamp: {timestamp}")
            print(f"Pedestrian analysis CSV: {self.ped_csv}")
            print(f"Vehicle analysis CSV: {self.vehicle_csv}")
            print(f"Vehicle counts CSV: {self.vcomp_csv}")
            print(f"Vehicle speeds CSV: {self.vspeed_csv}")
            
            # Initialize external processors with shared CSV paths
            self.crossing_analyzer = PedestrianCrossingAnalyzer(self.ped_csv)
            self.speed_detector = PedestrianSpeedDetector(pixels_per_meter, self.ped_csv)
            self.car_detector = CarPSMDetector(model_path, self.vehicle_csv)
            self.vehicle_analyzer = VehicleAnalyzer(self.vcomp_csv)
            self.vehicle_speed_detector = VehicleSpeedDetector(self.vspeed_csv)
            
            self.car_detector.set_pipeline(self)  # Pass pipeline reference to CARPSM detector
            
            # Detection confidence threshold
            self.conf_threshold = conf_threshold
            
            print("Pipeline initialization complete.")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def calculate_bbox_features(self, frame, bbox):
        """Calculate simple features for a bounding box"""
        x1, y1, x2, y2 = map(int, bbox[:4])
        if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
            return None
        try:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            # Calculate average color and shape features
            avg_color = np.mean(roi, axis=(0, 1))
            aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
            area = (x2 - x1) * (y2 - y1)
            return np.array([*avg_color, aspect_ratio, area])
        except Exception:
            return None

    def match_features(self, current_features, history_features, threshold=0.85):
        """Match current features with historical features"""
        if current_features is None or history_features is None:
            return False
        try:
            # Normalize features
            current_norm = np.linalg.norm(current_features)
            history_norm = np.linalg.norm(history_features)
            if current_norm == 0 or history_norm == 0:
                return False
            current_features = current_features / current_norm
            history_features = history_features / history_norm
            # Calculate similarity
            similarity = np.dot(current_features, history_features)
            return similarity > threshold
        except Exception:
            return False

    def process_detections(self, results, frame):
        """Process YOLO detections and return tracked objects"""
        # Separate pedestrian and vehicle detections
        ped_detections = []
        vehicle_detections = []
        
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            if conf > self.conf_threshold:
                if int(cls) == 3:  # Pedestrian class
                    ped_detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
                elif int(cls) in [0, 1, 2, 4, 5, 6]:  # Valid vehicle classes only
                    vehicle_detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
                    # Store the original class for this detection
                    bbox_key = (float(x1), float(y1), float(x2), float(y2))
                    self.temp_vehicle_classes[bbox_key] = int(cls)
        
        # Update tracker for pedestrians
        if len(ped_detections) > 0:
            ped_tracks = self.ped_tracker.update(np.array(ped_detections))
            
            # Process each track for consistent IDs
            for track in ped_tracks:
                tracker_id = int(track[4])
                
                # Get or assign global ID
                if tracker_id not in self.ped_id_mapping:
                    self.ped_id_mapping[tracker_id] = self.next_ped_id
                    self.next_ped_id += 1
                
                # Update track with global ID
                track[4] = self.ped_id_mapping[tracker_id]
                
                # Update appearance history
                current_features = self.calculate_bbox_features(frame, track)
                if current_features is not None:
                    if track[4] not in self.appearance_history:
                        self.appearance_history[track[4]] = []
                    self.appearance_history[track[4]].append((track[:4], current_features))
                    # Keep only last 5 appearances
                    self.appearance_history[track[4]] = self.appearance_history[track[4]][-5:]
        else:
            ped_tracks = np.empty((0, 5))
            
        # Update tracker for vehicles
        if len(vehicle_detections) > 0:
            vehicle_tracks = self.vehicle_tracker.update(np.array(vehicle_detections))
            
            # Process each vehicle track
            for track in vehicle_tracks:
                tracker_id = int(track[4])
                bbox = track[:4]
                
                # Get or assign global ID
                if tracker_id not in self.vehicle_id_mapping:
                    self.vehicle_id_mapping[tracker_id] = self.next_vehicle_id
                    self.next_vehicle_id += 1
                
                # Update track with global ID
                global_id = self.vehicle_id_mapping[tracker_id]
                track[4] = global_id
                
                # Find the closest matching detection to get its class
                closest_bbox = None
                min_dist = float('inf')
                track_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                
                for det_bbox in self.temp_vehicle_classes:
                    det_center = ((det_bbox[0] + det_bbox[2])/2, (det_bbox[1] + det_bbox[3])/2)
                    dist = ((track_center[0] - det_center[0])**2 + 
                           (track_center[1] - det_center[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_bbox = det_bbox
                
                # Update vehicle class if we found a close match
                if closest_bbox and min_dist < 50:  # 50 pixel threshold
                    self.vehicle_classes[global_id] = self.temp_vehicle_classes[closest_bbox]
        else:
            vehicle_tracks = np.empty((0, 5))
            
        # Clear temporary vehicle classes for next frame
        self.temp_vehicle_classes = {}
            
        return ped_tracks, vehicle_tracks

    def process_video(self, video_path: str):
        """Process video file and coordinate external processing"""
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        try:
            print(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties and setup
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Get first frame for setup
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")
            
            # Select lanes and target y-coordinate for car detection
            self.crossing_analyzer.select_points(first_frame)
            self.car_detector.lanes = self.crossing_analyzer.lanes
            self.car_detector.select_target_y(first_frame)
            
            # Select lines for vehicle speed detection
            self.vehicle_speed_detector.select_lines(first_frame)
            
            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create output directory and video writers
            os.makedirs('output', exist_ok=True)
            ped_output_path = os.path.join('output', f"processed_{os.path.basename(video_path)}")
            car_output_path = os.path.join('output', f"carpsm_{os.path.basename(video_path)}")
            vcomp_output_path = os.path.join('output', f"vcomp_{os.path.basename(video_path)}")
            vspeed_output_path = os.path.join('output', f"vspeed_{os.path.basename(video_path)}")
            
            ped_out = cv2.VideoWriter(ped_output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            car_out = cv2.VideoWriter(car_output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            vcomp_out = cv2.VideoWriter(vcomp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            vspeed_out = cv2.VideoWriter(vspeed_output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = datetime.now().strftime('%H:%M:%S')
                frame_time = float(frame_count) / float(self.fps)  # Ensure float division
                
                try:
                    # Get detections from YOLO model with NMS
                    results = self.model(frame, conf=self.conf_threshold, iou=0.5)[0]
                    
                    # Process detections and get tracked objects
                    ped_tracks, vehicle_tracks = self.process_detections(results, frame)
                    
                    # Process frames through external components with tracked objects
                    ped_frame = self.crossing_analyzer.process_frame(frame.copy(), ped_tracks, frame_time)
                    ped_frame = self.speed_detector.process_frame(ped_frame, ped_tracks, frame_time)
                    car_frame = self.car_detector.process_frame(frame.copy(), vehicle_tracks, current_time)
                    vcomp_frame = self.vehicle_analyzer.process_frame(frame.copy(), vehicle_tracks, self.vehicle_classes)
                    
                    # Process vehicle speed detection with vehicle tracks
                    vspeed_frame = self.vehicle_speed_detector.process_frame(frame.copy(), vehicle_tracks)
                    
                    # Add timestamp to frames
                    for frame_to_write in [ped_frame, car_frame, vcomp_frame, vspeed_frame]:
                        cv2.putText(frame_to_write, f"Time: {current_time}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Write frames
                    ped_out.write(ped_frame)
                    car_out.write(car_frame)
                    vcomp_out.write(vcomp_frame)
                    vspeed_out.write(vspeed_frame)
                    
                    # Display frames
                    combined_frame = np.hstack((ped_frame, car_frame))
                    cv2.imshow('Traffic Analysis', combined_frame)
                    cv2.imshow('Vehicle Composition', vcomp_frame)
                    cv2.imshow('Vehicle Speed', vspeed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing interrupted by user")
                        break
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    # Write original frame if processing fails
                    ped_out.write(frame)
                    car_out.write(frame)
                    vcomp_out.write(frame)
                    vspeed_out.write(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
            # Save final results
            print("Saving final results...")
            self.vehicle_analyzer.save_results_to_csv()
            self.vehicle_analyzer.print_statistics()
            self.vehicle_speed_detector.save_results_to_csv()
            self.vehicle_speed_detector.print_statistics()
            
            print(f"\nProcessing complete!")
            print(f"Pedestrian output saved to: {ped_output_path}")
            print(f"Car PSM output saved to: {car_output_path}")
            print(f"Vehicle Composition output saved to: {vcomp_output_path}")
            print(f"Vehicle Speed output saved to: {vspeed_output_path}")
            print(f"CSV files saved to logs directory:")
            print(f"  - Pedestrian analysis: {os.path.abspath(self.ped_csv)}")
            print(f"  - Vehicle analysis: {os.path.abspath(self.vehicle_csv)}")
            print(f"  - Vehicle counts: {os.path.abspath(self.vcomp_csv)}")
            print(f"  - Vehicle speeds: {os.path.abspath(self.vspeed_csv)}")
            
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            raise
            
        finally:
            if 'cap' in locals():
                cap.release()
            if 'ped_out' in locals():
                ped_out.release()
            if 'car_out' in locals():
                car_out.release()
            if 'vcomp_out' in locals():
                vcomp_out.release()
            if 'vspeed_out' in locals():
                vspeed_out.release()
            cv2.destroyAllWindows()

def main():
    try:
        # Configuration
        model_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\best(3).pt"
        video_path = r"C:\Ahsan\FYP System\Pipeline Scripts\Ped Work\export6.mp4"
        pixels_per_meter = 100  # Adjust this value based on your video's scale
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        
        pipeline = PedestrianPipeline(model_path, pixels_per_meter)
        pipeline.process_video(video_path)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return

if __name__ == "__main__":
    main()
