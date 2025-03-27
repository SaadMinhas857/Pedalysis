import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

class PedestrianCrosswalkDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Store pedestrian y-coordinates
        self.pedestrian_y_coords = []
        
        # Statistics
        self.mean_y = None
        self.std_y = None
        self.last_frame = None
    
    def process_video(self, video_path: str, output_path: str = 'output.mp4'):
        """
        Process video file and detect pedestrians
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Initialize video writer with mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (frame_width, frame_height)
            )
            
            frame_count = 0
            print("Processing video... Press 'q' to stop")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Store last frame for final visualization
                self.last_frame = frame.copy()
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write frame to output video
                output_video.write(processed_frame)
                
                # Display live processing
                cv2.imshow('Processing Video', processed_frame)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing stopped by user")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
            # Calculate statistics
            self.calculate_statistics()
            
            # Create final visualization
            final_frame = self.create_final_visualization()
            
            # Save final frame
            final_frame_path = output_path.rsplit('.', 1)[0] + '_final.jpg'
            cv2.imwrite(final_frame_path, final_frame)
            
            # Display final frame
            cv2.imshow('Final Result', final_frame)
            cv2.waitKey(0)
            
            # Cleanup
            cap.release()
            output_video.release()
            cv2.destroyAllWindows()
            
            print(f"\nProcessing complete!")
            print(f"Output video saved to: {output_path}")
            print(f"Final frame saved to: {final_frame_path}")
            print(f"\nStatistics:")
            print(f"Average y-coordinate: {self.mean_y:.2f}")
            print(f"Standard deviation: {self.std_y:.2f}")
            
        except cv2.error as e:
            print(f"OpenCV Error: {str(e)}")
            raise
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise
        finally:
            # Ensure cleanup happens even if there's an error
            try:
                cap.release()
                output_video.release()
                cv2.destroyAllWindows()
            except:
                pass
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with detections
        """
        # Run YOLO detection
        results = self.model(frame)[0]
        
        # Process detections
        processed_frame = frame.copy()
        
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            
            # Check if detection is a pedestrian (class 3) with sufficient confidence
            if int(cls) == 3 and conf >= self.conf_threshold:
                # Calculate center y-coordinate of bounding box
                center_y = (y1 + y2) / 2
                self.pedestrian_y_coords.append(center_y)
                
                # Draw bounding box
                cv2.rectangle(
                    processed_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                
                # Add confidence label
                label = f"Pedestrian: {conf:.2f}"
                cv2.putText(
                    processed_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        return processed_frame
    
    def calculate_statistics(self):
        """Calculate mean and standard deviation of y-coordinates"""
        if self.pedestrian_y_coords:
            self.mean_y = np.mean(self.pedestrian_y_coords)
            self.std_y = np.std(self.pedestrian_y_coords)
        else:
            print("Warning: No pedestrians detected in video")
            self.mean_y = 0
            self.std_y = 0
    
    def create_final_visualization(self) -> np.ndarray:
        """
        Create final visualization with average line and standard deviation bounds
        
        Returns:
            Frame with visualization
        """
        if self.last_frame is None:
            raise ValueError("No frames processed yet")
        
        final_frame = self.last_frame.copy()
        h, w = final_frame.shape[:2]
        
        # Draw average line (green)
        cv2.line(final_frame, (0, int(self.mean_y)), (w, int(self.mean_y)), (0, 255, 0), 2)
        
        # Draw standard deviation bounds (red)
        cv2.line(final_frame, (0, int(self.mean_y - self.std_y)), (w, int(self.mean_y - self.std_y)), (0, 0, 255), 2)
        cv2.line(final_frame, (0, int(self.mean_y + self.std_y)), (w, int(self.mean_y + self.std_y)), (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(final_frame, "Average Crosswalk Line", (10, int(self.mean_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(final_frame, "±1 Std Dev", (10, int(self.mean_y - self.std_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return final_frame

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Pedestrian Crosswalk Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    
    # Initialize detector
    detector = PedestrianCrosswalkDetector(args.model, args.conf)
    
    # Process video
    try:
        detector.process_video(args.video, args.output)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main() 