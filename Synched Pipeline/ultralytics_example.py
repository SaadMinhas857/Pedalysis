import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

# Adjust these coordinates based on your specific road area
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

# Real-world dimensions (17m width)
REAL_WIDTH = 17  # meters

REAL_HEIGHT = 30 # meters

# Calculate target dimensions to maintain real-world proportions
PIXELS_PER_METER = 50  # This gives us good visualization while maintaining proportions
TARGET_WIDTH = int(REAL_WIDTH * PIXELS_PER_METER)   # 17m * 40px/m = 680px
TARGET_HEIGHT = int(REAL_HEIGHT * PIXELS_PER_METER)  # 80m * 40px/m = 3200px

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.4,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.5, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--select_roi", 
        action="store_true",
        help="Enable interactive selection of ROI points"
    )
    parser.add_argument(
        "--extend_frame",
        action="store_true",
        help="Enable extended frame for point selection"
    )

    return parser.parse_args()


def create_extended_frame(frame, extension=500):
    """Create an extended frame with black borders, keeping original frame centered"""
    h, w = frame.shape[:2]
    extended_h = h + 2 * extension
    extended_w = w + 2 * extension
    
    # Create black canvas
    extended_frame = np.zeros((extended_h, extended_w, 3), dtype=np.uint8)
    
    # Calculate center position
    center_y = (extended_h - h) // 2
    center_x = (extended_w - w) // 2
    
    # Place original frame in center
    extended_frame[center_y:center_y+h, center_x:center_x+w] = frame
    
    # Draw frame boundary
    cv2.rectangle(extended_frame, (center_x, center_y), 
                 (center_x+w, center_y+h), (0, 255, 0), 2)
    
    return extended_frame, center_x, center_y


def select_roi(first_frame, extend_frame=False):
    """Allow user to select region of interest"""
    points = []
    frame_extension = 500  # pixels to extend frame
    
    if extend_frame:
        frame, center_x, center_y = create_extended_frame(first_frame, frame_extension)
        print(f"Frame extended by {frame_extension} pixels in each direction")
    else:
        frame = first_frame.copy()
        center_x, center_y = 0, 0
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert extended coordinates back to original frame coordinates
            orig_x = x - center_x
            orig_y = y - center_y
            points.append((orig_x, orig_y))
            
            # Draw on extended frame
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                prev_x = points[-2][0] + center_x
                prev_y = points[-2][1] + center_y
                cv2.line(frame, (prev_x, prev_y), (x, y), (0, 255, 0), 2)
            if len(points) == 4:
                first_x = points[0][0] + center_x
                first_y = points[0][1] + center_y
                cv2.line(frame, (x, y), (first_x, first_y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", frame)
    
    # Create window and center it on screen
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    
    # Get screen dimensions
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        # Fallback to common resolution if tkinter is not available
        screen_width = 1920
        screen_height = 1080
    
    # Calculate window size (80% of screen size)
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    cv2.resizeWindow("Select ROI", window_width, window_height)
    
    # Calculate window position
    pos_x = (screen_width - window_width) // 2
    pos_y = (screen_height - window_height) // 2
    cv2.moveWindow("Select ROI", pos_x, pos_y)
    
    cv2.setMouseCallback("Select ROI", mouse_callback)
    
    print("\nInstructions:")
    print("1. Click to select 4 points to define the road area")
    print("2. Points can be selected outside the original frame (green rectangle)")
    print("3. Press 'q' when done or 'r' to reset")
    print("4. Press 'e' to toggle between extended and original frame view")
    
    frame_copy = frame.copy()
    show_extended = True
    
    while True:
        cv2.imshow("Select ROI", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') and len(points) == 4:
            break
        elif key == ord('r'):
            points = []
            frame = frame_copy.copy()
            cv2.imshow("Select ROI", frame)
        elif key == ord('e') and extend_frame:
            show_extended = not show_extended
            if show_extended:
                frame = frame_copy.copy()
            else:
                # Show original frame only
                frame = first_frame.copy()
                # Draw existing points
                for i, (x, y) in enumerate(points):
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(frame, points[i-1], (x, y), (0, 255, 0), 2)
                    if i == 3:
                        cv2.line(frame, points[3], points[0], (0, 255, 0), 2)
            cv2.imshow("Select ROI", frame)
    
    cv2.destroyAllWindows()
    return np.array(points)


if __name__ == "__main__":
    args = parse_arguments()

    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    
    # Use your custom model
    model = YOLO("best(3).pt")
    
    # Get first frame for ROI selection if enabled
    if args.select_roi:
        cap = cv2.VideoCapture(args.source_video_path)
        ret, first_frame = cap.read()
        if ret:
            SOURCE = select_roi(first_frame, args.extend_frame)
            print(f"Selected ROI points: {SOURCE}")
        cap.release()

    # Initialize tracker with correct parameters
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps * 2))  # 2 seconds history

    # Create debug window to show transformed view
    debug_frame = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) * 255

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame_idx, frame in enumerate(frame_generator):
            # Skip the first 100 frames
            
            
            # Run detection
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Filter detections by confidence
            detections = detections[detections.confidence > args.confidence_threshold]
            
            # Process only objects inside the defined polygon
            detections = detections[polygon_zone.trigger(detections)]
            
            # Apply NMS filtering
            detections = detections.with_nms(threshold=args.iou_threshold)
            
            # Update tracker
            detections = byte_track.update_with_detections(detections=detections)

            # Get bottom center points of all detections
            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            
            # Transform points using perspective transformation
            transformed_points = view_transformer.transform_points(points=points).astype(int)

            # Clear debug frame and create white background
            debug_frame.fill(255)
            
            # Draw grid with better visibility
            # Major grid lines (every 5 meters)
            for x in range(0, TARGET_WIDTH, 5 * PIXELS_PER_METER):
                cv2.line(debug_frame, (x, 0), (x, TARGET_HEIGHT), (200, 200, 200), 2)
                # Add distance markers in meters
                meters = x / PIXELS_PER_METER
                cv2.putText(debug_frame, f"{meters:.1f}m", (x+2, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            for y in range(0, TARGET_HEIGHT, 5 * PIXELS_PER_METER):
                cv2.line(debug_frame, (0, y), (TARGET_WIDTH, y), (200, 200, 200), 2)
                # Add distance markers in meters
                meters = y / PIXELS_PER_METER
                cv2.putText(debug_frame, f"{meters:.1f}m", (5, y+15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # Minor grid lines (every 1 meter)
            for x in range(0, TARGET_WIDTH, PIXELS_PER_METER):
                cv2.line(debug_frame, (x, 0), (x, TARGET_HEIGHT), (230, 230, 230), 1)
            for y in range(0, TARGET_HEIGHT, PIXELS_PER_METER):
                cv2.line(debug_frame, (0, y), (TARGET_WIDTH, y), (230, 230, 230), 1)

            # Draw border
            cv2.rectangle(debug_frame, (0, 0), (TARGET_WIDTH-1, TARGET_HEIGHT-1), (0, 0, 0), 2)

            # Process each detection for transformed view
            for tracker_id, [x, y], class_id, confidence in zip(
                detections.tracker_id, transformed_points, detections.class_id, detections.confidence
            ):
                # Skip if point is outside the target area
                if not (0 <= x < TARGET_WIDTH and 0 <= y < TARGET_HEIGHT):
                    continue
                    
                # Record y-coordinate (for vertical movement tracking)
                coordinates[tracker_id].append(y)
                
                # Draw vehicle position with better visibility
                cv2.circle(debug_frame, (x, y), 5, (0, 0, 255), -1)  # Red dot for current position
                cv2.putText(debug_frame, f"#{tracker_id}", (x+5, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw trails with fading effect
                points_list = list(coordinates[tracker_id])
                if len(points_list) >= 2:
                    for i in range(1, len(points_list)):
                        # Calculate alpha for fading effect
                        alpha = 0.5 * (i / len(points_list))
                        # Estimate x position based on position in array
                        prev_x = max(0, min(TARGET_WIDTH-1, x - (len(points_list) - i)))
                        curr_x = max(0, min(TARGET_WIDTH-1, x - (len(points_list) - i - 1)))
                        # Draw trail with fading effect
                        cv2.line(debug_frame, 
                                (prev_x, points_list[i-1]), 
                                (curr_x, points_list[i]),
                                (0, 0, int(255 * alpha)), 2)

                # Calculate and display speed if enough history
                if len(points_list) >= video_info.fps / 2:
                    # Calculate speed based on actual distance in meters
                    start_y = points_list[0] / PIXELS_PER_METER  # Convert to meters
                    end_y = points_list[-1] / PIXELS_PER_METER   # Convert to meters
                    distance = abs(end_y - start_y)  # Distance in meters
                    time = len(points_list) / video_info.fps     # Time in seconds
                    speed = distance / time * 3.6                 # Convert to km/h
                    
                    # Display speed on transformed view
                    cv2.putText(debug_frame, f"{speed:.1f} km/h", (x+5, y+15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Calculate speeds and prepare labels for original frame
            labels = []
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                class_name = model.names[class_id]
                
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    # Not enough history for speed calculation
                    labels.append(f"#{tracker_id} {class_name}")
                else:
                    # Calculate speed based on actual distance in meters
                    points_list = list(coordinates[tracker_id])
                    start_y = points_list[0] / PIXELS_PER_METER  # Convert to meters
                    end_y = points_list[-1] / PIXELS_PER_METER   # Convert to meters
                    distance = abs(end_y - start_y)  # Distance in meters
                    time = len(points_list) / video_info.fps     # Time in seconds
                    speed = distance / time * 3.6                 # Convert to km/h
                    
                    labels.append(f"#{tracker_id} {class_name} {speed:.1f} km/h")

            # Add frame number to original frame
            cv2.putText(
                frame, 
                f"Frame: {frame_idx} | FPS: {video_info.fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                text_scale, 
                (0, 255, 0), 
                thickness
            )

            # Annotate the frame
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # Draw polygon zone
            polygon_points = polygon_zone.polygon.reshape(-1, 2)
            sv.draw_polygon(scene=annotated_frame, polygon=polygon_points, color=sv.Color.RED)

            # Add title and legend to debug frame
            cv2.putText(debug_frame, "Bird's Eye View", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add scale information
            scale_text = f"Scale: {PIXELS_PER_METER}px = 1m"
            cv2.putText(debug_frame, scale_text, (10, TARGET_HEIGHT-20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Resize debug frame for display (keep aspect ratio)
            display_height = 800
            display_width = int(TARGET_WIDTH * (display_height / TARGET_HEIGHT))
            debug_resized = cv2.resize(debug_frame, (display_width, display_height), 
                                     interpolation=cv2.INTER_AREA)
            
            # Write frame to output video
            sink.write_frame(annotated_frame)
            
            # Show frames
            cv2.imshow("Detection", annotated_frame)
            cv2.imshow("Transformed View", debug_resized)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        cv2.destroyAllWindows()
