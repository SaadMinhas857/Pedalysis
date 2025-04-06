import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

class ObjectTracking:
    def __init__(self, model_path="best(3).pt", im0=None, w=None, h=None, fps=None):
        """
        Initialize the object tracking class.
        
        Args:
            model_path (str): Path to the YOLOv8 model.
            im0 (np.array, optional): The input frame.
            w (int, optional): The width of the frame.
            h (int, optional): The height of the frame.
            fps (int, optional): Frames per second.
        """
        self.model = YOLO(model_path)  # Load the YOLOv8 model
        self.frame_size = (w, h) if w and h else (None, None)
        self.frames_per_second = fps  # Use passed FPS
        self.line_points = []  # Store clicked points for reference lines
        self.ref_line_a = []  # First reference line
        self.ref_line_b = []  # Second reference line
        self.trk_history = {}  # Store tracking history
        self.cross_times = {}  # Store crossing times for vehicles
        self.wait_for_reference_lines = True  # Flag to wait for user input
        self.mouse_callback(im0)  # Set up mouse callback for reference lines
        self.im0 = im0

    def extract_tracks(self, results):
        """
        Extracts bounding boxes, class labels, and track IDs from YOLOv8 tracking results.
        """
        self.boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        self.clss = results[0].boxes.cls.cpu().numpy()  # Class labels
        self.trk_ids = results[0].boxes.id.cpu().numpy()  # Tracking IDs

    def calculate_bbox_center(self, box):
        """
        Calculate the center of a bounding box.
        """
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        return (x_center, y_center)

    def store_track_info(self, trk_id, box):
        """
        Store tracking information (center of bounding box) for the object.
        """
        if trk_id not in self.trk_history:
            self.trk_history[trk_id] = {"track": []}
        bbox_center = self.calculate_bbox_center(box)
        self.trk_history[trk_id]["track"].append(bbox_center)

        # Limit the number of points stored to 30
        if len(self.trk_history[trk_id]["track"]) > 30:
            self.trk_history[trk_id]["track"].pop(0)

        return self.trk_history[trk_id]["track"]

    def store_tracking_data(self, results):
        """ Store tracking data for each detected object. """
        for bbox, cls_id, track_id in zip(self.boxes, self.clss, self.trk_ids):
            if cls_id == 4:  # Filter for class 4 (Sedan)
                self.store_track_info(track_id, bbox)

    def set_reference_lines(self, points_line_a, points_line_b):
        """
        Set reference lines for speed calculation based on user input.
        """
        self.ref_line_a = points_line_a
        self.ref_line_b = points_line_b

    def capture_crossing_time(self, track_id, object_center, current_time):
        """
        Capture the time when an object crosses Line A or Line B.
        """
        y_center = object_center[1]
        if track_id not in self.cross_times:
            self.cross_times[track_id] = {"line_a": None, "line_b": None}

        if self.ref_line_a and self.ref_line_b:
            line_a_y = (self.ref_line_a[0][1] + self.ref_line_a[1][1]) / 2
            line_b_y = (self.ref_line_b[0][1] + self.ref_line_b[1][1]) / 2

            if self.cross_times[track_id]["line_a"] is None and y_center > line_a_y:
                self.cross_times[track_id]["line_a"] = current_time

            if self.cross_times[track_id]["line_b"] is None and y_center > line_b_y:
                self.cross_times[track_id]["line_b"] = current_time

    def calculate_speed(self, track_id):
        """
        Calculate speed based on the time difference and pixel-to-distance conversion.
        """
        if track_id in self.cross_times:
            crossing_times = self.cross_times[track_id]
            if crossing_times["line_a"] is not None and crossing_times["line_b"] is not None:
                # Calculate time difference between crossing Line A and Line B
                time_diff = (crossing_times["line_b"] - crossing_times["line_a"]).total_seconds()
                if time_diff > 0:
                    # Calculate distance in pixels (between reference lines)
                    line_a_y = (self.ref_line_a[0][1] + self.ref_line_a[1][1]) / 2
                    line_b_y = (self.ref_line_b[0][1] + self.ref_line_b[1][1]) / 2
                    pixel_distance = abs(line_b_y - line_a_y)

                    # Convert to real-world distance (meters)
                    distance = pixel_distance * 0.05  # Example: 0.05 meters per pixel

                    # Calculate speed (distance/time)
                    speed = distance / time_diff
                    return speed
        return None

    def plot_bbox(self, frame, track_id, bbox, speed):
        """
        Annotate frame with detected vehicle, tracking line, and speed.
        """
        x1, y1, x2, y2 = map(int, bbox)
        label = f"ID: {track_id} Speed: {speed:.2f} m/s"
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw reference lines
        if self.ref_line_a:
            cv2.line(frame, self.ref_line_a[0], self.ref_line_a[1], (0, 0, 255), 2)
        if self.ref_line_b:
            cv2.line(frame, self.ref_line_b[0], self.ref_line_b[1], (0, 0, 255), 2)

    def mouse_callback(self, im0):
        """
        Set up mouse callback for capturing reference line points and drawing them.
        """
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.line_points) < 4:
                    self.line_points.append((x, y))  # Capture the point
                    cv2.circle(param, (x, y), 5, (0, 255, 0), -1)

                    # Draw lines between consecutive points
                    if len(self.line_points) > 1:
                        cv2.line(param, self.line_points[-2], self.line_points[-1], (255, 0, 0), 2)

                    # Once 4 points are clicked, set reference lines
                    if len(self.line_points) == 4:
                        self.ref_line_a = self.line_points[:2]
                        self.ref_line_b = self.line_points[2:]
                        self.wait_for_reference_lines = False
                        print("Reference lines set!")

                    cv2.imshow("Select Reference Lines", param)

        cv2.namedWindow("Select Reference Lines")
        cv2.setMouseCallback("Select Reference Lines", on_mouse_click)

    def process_video(self, results, im0, current_time, w, h, fps):
        """
        Process video and wait for user input for reference lines.
        """
        while self.wait_for_reference_lines:
            cv2.imshow("Select Reference Lines", im0)
            cv2.waitKey(1)  # Just wait for the mouse click to set reference lines

        self.extract_tracks(results)  # Extract tracks from results
        self.store_tracking_data(results)  # Store tracking info for each vehicle

        # Process each tracked vehicle and calculate speed
        for trk_id in self.trk_history.keys():
            speed = self.calculate_speed(trk_id)
            if speed is not None:
                # Annotate the frame with tracking line and speed
                self.plot_bbox(im0, trk_id, self.trk_history[trk_id]["track"][-1], speed)

        return im0  # Return the updated frame
