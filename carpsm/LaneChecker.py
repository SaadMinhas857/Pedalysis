import cv2
import numpy as np
import csv
from collections import defaultdict
from datetime import datetime
from ultralytics.utils.plotting import Annotator, colors

class LaneChecker:
    def __init__(self, lane_regions, csv_file="lane_log.csv", y_target=400, tolerance=5):
        """
        Initializes the LaneChecker with the provided lane regions and logging settings.

        Args:
            lane_regions (list): List of lane regions as [(x1, y1, x2, y2, x3, y3, x4, y4), ...]
            csv_file (str): File path for the CSV log file.
            y_target (int): Y-coordinate to check (e.g., 500).
            tolerance (int): Tolerance for Y-coordinate checking.
        """
        self.lane_indices = {} 
        self.lane_regions = lane_regions
        self.csv_file = csv_file
        self.y_target = y_target
        self.tolerance = tolerance
        self.trk_history = defaultdict(list)
        self.logged_vehicles = set()  # Set of vehicles that have already been logged

        # Initialize the CSV file if it doesn't exist
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Time", "Lane"])

    def store_track_info(self, track_id, box):
        """
        Stores track data (center points) for a given track ID.

        Args:
            track_id (int): Object track ID.
            box (list): Object bounding box data.

        Returns:
            list: Updated tracking history for the given track_id.
        """
        track = self.trk_history[track_id]
        bbox_center = self.get_bbox_center(box)
        track.append(bbox_center)

        # Keep track history manageable by removing old points if it exceeds 30 points
        if len(track) > 30:
            track.pop(0)

        return track

    def get_bbox_center(self, bbox):
        """
        Calculate the center of the bounding box.

        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            tuple: Center point (cx, cy).
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    def check_y_coordinate(self, track_id, bbox_center):
        """
        Checks if the bounding box center is within the target Y-coordinate range.

        Args:
            track_id (int): Object track ID.
            bbox_center (tuple): Center coordinates (cx, cy) of the bounding box.

        Returns:
            bool: True if the Y-coordinate is within the target Y ± tolerance.
        """
        cx, cy = bbox_center
        if abs(cy - self.y_target) <= self.tolerance:
            return True
        return False

    def check_lane(self, track_id, bbox_center):
        """
        Check in which lane the bounding box center lies.

        Args:
            track_id (int): The ID of the object being tracked.
            bbox_center (tuple): Center coordinates (cx, cy) of the bounding box.

        Returns:
            int: Index of the lane the bounding box center lies in, or -1 if not in any lane.
        """
        for idx, lane in enumerate(self.lane_regions):
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = lane
            if self.is_inside_region(bbox_center, x1, y1, x2, y2, x3, y3, x4, y4):
                return idx
        return -1

    def is_inside_region(self, point, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Checks if the point is inside the polygon formed by four corner points of the lane.

        Args:
            point (tuple): Point coordinates (cx, cy).
            x1, y1, x2, y2, x3, y3, x4, y4: Coordinates of the lane region.

        Returns:
            bool: True if the point is inside the lane, otherwise False.
        """
        cx, cy = point
        polygon = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)
        return cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False) >= 0

    def log_lane_entry(self, track_id, lane_idx, log_time):
        """
        Logs the lane entry details to a CSV file.

        Args:
            track_id (int): Object track ID.
            lane_idx (int): Index of the lane the object is in.
            log_time (str): Time passed from the SpeedEstimator class.
        """
        if track_id in self.logged_vehicles:
            return

        # Add vehicle ID to logged_vehicles
        self.logged_vehicles.add(track_id)

        # Append the data to the CSV file
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([track_id, log_time, lane_idx])
    def get_persistent_lane_index(self, track_id):
        """
        Retrieves the persistent lane index for a track ID, if available.

        Args:
            track_id (int): The ID of the tracked object.

        Returns:
            int: The lane index, or -1 if not determined.
        """
        return self.lane_indices.get(track_id, -1)
    def plot_box_and_track(self, im0, track_id, box, cls, track, lane_idx, annotator, class_label):
        """
        Plots the bounding box, track, and lane information on the image.

        Args:
            im0: Frame on which to plot.
            track_id: ID of the tracked object.
            box: Bounding box coordinates [x1, y1, x2, y2].
            cls: Object class ID.
            track: List of previous center points for the track.
            lane_idx: Lane index of the object.
            annotator: Annotator object to draw bounding boxes.
            class_label: Label for the detected class.
        """
        lane_label = f"Lane {lane_idx}" if lane_idx != -1 else "Unknown Lane"
        bbox_color = colors(int(track_id)) if track_id is not None else (255, 0, 255)

        # Convert box coordinates to integers
        x1, y1, x2, y2 = [int(coord) for coord in box]

        # Draw the bounding box and label (class + lane)
        annotator.box_label([x1, y1, x2, y2], f"{class_label}, {lane_label}", color=bbox_color, txt_color=(255, 255, 255))

        # Draw the tracking path
        if len(track) > 1:
            pts = np.array(track, np.int32).reshape((-1, 1, 2))
            cv2.polylines(im0, [pts], isClosed=False, color=bbox_color, thickness=2)

        # Draw the last point (current position) as a circle
        cv2.circle(im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)
