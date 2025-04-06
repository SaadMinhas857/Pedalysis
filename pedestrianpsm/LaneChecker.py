import cv2
import numpy as np
import csv
from datetime import datetime, timedelta
from ultralytics.utils.plotting import Annotator, colors

class LaneChecker:
    def __init__(self, lane_regions, lane_centers, csv_file="lane_log.csv", center_tolerance=10):
        """
        Initializes the LaneChecker with lane regions and lane centers.

        Args:
            lane_regions (list): List of lane regions (polygons) where each lane is represented as a tuple of 4 points.
            lane_centers (list): List of center points for each lane (based on X-coordinate).
            csv_file (str): File path for logging lane entry, exit, and middle crossing times.
            center_tolerance (int): Tolerance in pixels for detecting when the center is crossed.
        """
        self.lane_regions = lane_regions
        self.lane_centers = lane_centers
        self.csv_file = csv_file
        self.center_tolerance = center_tolerance
        self.tracked_objects = {}  # Dictionary to store tracked objects' state
        self.init_csv()

    def init_csv(self):
        """Initializes the CSV file with headers for lane entry/exit and center crossing times."""
        header = ["Track ID"]
        for i in range(1, len(self.lane_regions) + 1):
            header.append(f"Lane {i} Entry")
            header.append(f"Lane {i} Exit")
            header.append(f"Lane {i} Mid Cross")  # New column for center crossing time

        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def log_event(self, event_type, lane_number, trk_id, current_time):
        """
        Logs entry, exit, or center crossing events for the object.

        Args:
            event_type (str): Type of event ("enter", "exit", "mid").
            lane_number (int): Lane number being logged.
            trk_id (int): The tracking ID of the object.
            current_time (str): The current time when the event occurs.
        """
        lane_index = (lane_number - 1) * 3
        if event_type == "enter":
            column = lane_index + 1
        elif event_type == "exit":
            column = lane_index + 2
        else:  # "mid"
            column = lane_index + 3

        rows = []
        with open(self.csv_file, mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        found_row = False
        for row in rows:
            if row[0] == str(trk_id):  # Track ID exists
                row[column] = current_time
                found_row = True
                break
        
        if not found_row:
            new_row = [None] * (len(self.lane_regions) * 3 + 1)
            new_row[0] = str(trk_id)
            new_row[column] = current_time
            rows.append(new_row)

        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def calculate_bbox_center(self, box):
        """
        Calculates the geometric center of the bounding box.
        
        Args:
            box (list): Bounding box coordinates [x1, y1, x2, y2].
            
        Returns:
            tuple: (center_x, center_y) coordinates of the bounding box center.
        """
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2  # Use float division for more precision
        center_y = (y1 + y2) / 2
        return (center_x, center_y)

    def store_track_info(self, trk_id, box):
        """
        Stores the tracking information (center of the bounding box) for the object.
        
        Args:
            trk_id (int): Tracking ID of the object.
            box (list): Bounding box coordinates [x1, y1, x2, y2].
        """
        track = self.tracked_objects.setdefault(trk_id, {"track": [], "events": [], "entry_time": None, "current_lane": -1})
        bbox_center = self.calculate_bbox_center(box)
        track["track"].append(bbox_center)

        # Limit the number of points stored to 30
        if len(track["track"]) > 30:
            track["track"].pop(0)

        return track["track"]

    def check_crossing(self, trk_id, box, current_time, bbox_center):
        """
        Checks if the object has entered or exited a lane and logs midpoint crossing.

        Args:
            trk_id (int): Tracking ID of the object.
            box (list): Bounding box coordinates [x1, y1, x2, y2].
            current_time (str): Current timestamp for logging.
            bbox_center (tuple): The center of the bounding box.
        """
        MIN_DELAY = timedelta(seconds=0.01)

        # Ensure the current_state has all necessary keys and default values
        current_state = self.tracked_objects.get(trk_id, {
            "current_lane": -1,  # Default to no lane assigned
            "entry_time": None,
            "events": [],
        })

        current_lane = current_state["current_lane"]
        entry_time = current_state["entry_time"]
        events = current_state["events"]

        for i, lane in enumerate(self.lane_regions):
            lane_polygon = np.array(lane, np.int32).reshape((-1, 2))
            is_in_lane = cv2.pointPolygonTest(lane_polygon, bbox_center, False) >= 0

            # Detect lane entry (based on X-coordinate)
            if is_in_lane and current_lane != i and "enter" not in events:
                self.log_event("enter", i + 1, trk_id, current_time)
                current_state["current_lane"] = i
                current_state["entry_time"] = current_time
                events.append("enter")

            # Detect midpoint crossing (based on X-coordinate)
            if current_state["entry_time"] is not None and "enter" in events:
                lane_center = self.lane_centers[i]  # X-coordinate of lane center
                if bbox_center[0] >= lane_center[0] and "mid" not in events:
                    if current_time - current_state["entry_time"] >= MIN_DELAY:
                        self.log_event("mid", i + 1, trk_id, current_time)
                        events.append("mid")

            # Detect lane exit (only if midpoint has been crossed)
            if not is_in_lane and current_lane == i and "mid" in events:
                self.log_event("exit", i + 1, trk_id, current_time)
                current_state["current_lane"] = -1
                events.append("exit")

        self.tracked_objects[trk_id] = current_state  # Update tracked state

    def plot_box_and_track(self, im0, track_id, box, cls, track, annotator, class_label):
        """
        Plots the bounding box, track, and lane information on the image.

        Args:
            im0: Frame on which to plot.
            track_id: ID of the tracked object.
            box: Bounding box coordinates [x1, y1, x2, y2].
            cls: Object class ID.
            track: List of previous track points.
            annotator: YOLOv8 Annotator object.
            class_label: Label for the object's class.
        """
        for i, lane in enumerate(self.lane_regions):
            lane_polygon = np.array(lane, np.int32).reshape((-1, 1, 2))  # Convert to (N, 1, 2) shape for polylines
            cv2.polylines(im0, [lane_polygon], isClosed=True, color=(255, 0, 0), thickness=2)  # Draw the lane region

            # Plot the lane center
            lane_center = self.lane_centers[i]
            cv2.circle(im0, lane_center, 5, (0, 0, 255), -1)  # Draw a red dot at the lane center

            # Annotate the lane center with text
            cv2.putText(im0, f"Lane {i + 1}", (lane_center[0] - 20, lane_center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Only process class 3 (pedestrian)
        if cls != 3:
            return

        label = f'{track_id}: {class_label}'
        annotator.box_label(box, label, color=colors(cls, True))

        # Plot the tracking history for this track
        if len(track) >= 2:
            for i in range(len(track) - 1):
                pt1 = tuple(map(int, track[i]))
                pt2 = tuple(map(int, track[i + 1]))
                cv2.line(im0, pt1, pt2, (0, 255, 0), 2)

        # Plot lane polygons (regions)
        for i, lane in enumerate(self.lane_regions):
            lane_polygon = np.array(lane, np.int32).reshape((-1, 1, 2))  # Convert to (N, 1, 2) shape for polylines
            cv2.polylines(im0, [lane_polygon], isClosed=True, color=(255, 0, 0), thickness=2)  # Draw the lane region

            # Plot the lane center
            lane_center = self.lane_centers[i]
            cv2.circle(im0, lane_center, 5, (0, 0, 255), -1)  # Draw a red dot at the lane center

            # Annotate the lane center with text
            cv2.putText(im0, f"Lane {i + 1}", (lane_center[0] - 20, lane_center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Optionally, display the image with cv2.imshow() (if you're not already doing this elsewhere)
        # cv2.imshow("Frame", im0)
        # cv2.waitKey(1)  # This is important to make the image refresh in real-time
