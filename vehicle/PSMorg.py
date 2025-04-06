import numpy as np
import cv2
from datetime import datetime
from ultralytics.utils.plotting import Annotator, colors
from carpsm.LaneChecker import LaneChecker
from carpsm.VSpeed import SpeedEstimator
from vehicle.Speed_3 import SpeedExtractor  # Assuming SpeedExtractor is in 'speed_extractor.py'

class SpeedEstimator2:
    def __init__(self, names, lane_regions, current_time, line1_y=400, line2_y=500, line3_y=600, target_x=800, view_img=False, line_thickness=2, csv_file="lane_log.csv", fps=30):
        """
        Initializes the SpeedEstimator2 class with specific settings.

        Args:
            names (list): List of names corresponding to detected object classes.
            lane_regions (list): List of tuples defining the polygon regions of each lane.
            line1_y, line2_y, line3_y (int): Y-coordinates for the three lines where speed will be calculated.
            target_x (int): X-coordinate target for lane checking.
            view_img (bool): Flag to determine if the image should be displayed.
            line_thickness (int): Line thickness for annotations.
            csv_file (str): Path to CSV file for logging lane entries.
            fps (int): Frames per second for video processing.
        """
        self.names = names
        self.target_x = target_x
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.csv_file = csv_file
        self.fps = fps
        
        # Initialize SpeedExtractor for speed calculation at three locations (lines)
        self.speed_extractor = SpeedExtractor(names, line1_y, line2_y, line3_y, fps)
        
        self.boxes = None
        self.trk_ids = None
        self.clss = None
        self.dist_data = {}  # Dictionary to store the speed for each track
        self.speed_data = {}  # Initialize speed_data to store the previous positions and timestamps
    
    def extract_tracks(self, tracks):
        """
        Extracts bounding boxes, class labels, and track IDs from YOLO tracking data.

        Args:
            tracks (YOLO object): YOLO detection object containing bounding boxes, class labels, and IDs.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        self.clss = tracks[0].boxes.cls.cpu().numpy()  # Class labels
        self.trk_ids = tracks[0].boxes.id.cpu().numpy()  # Tracking IDs

    def Speed(self, im0, tracks, current_time, skipped_frames=0):
        """
        Processes video frames to check and log vehicle lanes based on tracking data and current_time.

        Args:
            im0 (np.ndarray): The current video frame.
            tracks: Tracking information from YOLO.
            current_time (datetime): The current time for timestamping log entries.
            skipped_frames (int): Number of frames skipped since the last call.
        """
        self.im0 = im0
        if not tracks or tracks[0].boxes.id is None:
            return im0  # If no tracks are detected or there are issues with detection IDs, return the frame as is

        self.extract_tracks(tracks)

        # Initialize the annotator for plotting bounding boxes and labels
        self.annotator = Annotator(self.im0, line_width=self.line_thickness)

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            trk_id = int(trk_id)
            if cls != 4:  # Filtering for Class 4 (Sedan) only
                continue

            # Store track info and get center coordinates of the bounding box
            track = self.speed_extractor.store_track_info(trk_id, box)
            bbox_center = track[-1]
            class_label = self.names[cls] if cls < len(self.names) else 'Unknown'

            # Retrieve the previous position and current position for speed calculation
            previous_position = self.speed_data.get(trk_id, {}).get('last_position', (bbox_center[0], bbox_center[1]))
            current_position = (bbox_center[0], bbox_center[1])

            # Calculate the speed using the `calculate_speed` method from `SpeedExtractor`
            speed = self.speed_extractor.calculate_speed(trk_id, previous_position, current_position, current_time)

            if speed is not None:
                # Update the speed data for the track ID
                self.speed_data[trk_id] = {'last_position': current_position, 'last_time': current_time}

                # Plot the bounding box, speed, and save the data to CSV
                self.im0 = self.speed_extractor.plot_box_and_save_to_csv(self.im0, trk_id, speed, (int(bbox_center[0]), int(bbox_center[1])), current_time)

        self.im0 = self.annotator.result()
        return self.im0
