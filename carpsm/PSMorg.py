import xml.etree.ElementTree as ET 
from datetime import datetime, timedelta
import numpy as np
import cv2
import os
import csv
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from carpsm.LaneChecker import LaneChecker
from carpsm.VSpeed import SpeedEstimator

class SpeedEstimator2:
    def __init__(self, names, lane_regions, target_x=800, view_img=False, line_thickness=2, csv_file="lane_log.csv"):
        """
        Initializes the SpeedEstimator2 class with specific settings.

        Args:
            names (list): List of names corresponding to detected object classes.
            lane_regions (list): List of tuples defining the polygon regions of each lane.
            target_x (int): X-coordinate target for lane checking.
            view_img (bool): Flag to determine if the image should be displayed.
            line_thickness (int): Line thickness for annotations.
            csv_file (str): Path to CSV file for logging lane entries.
        """
        self.names = names  # Object class names from YOLO model
        self.lane_checker = LaneChecker(lane_regions, csv_file)  # Initialize the LaneChecker with the specified lane regions and CSV file
        
        # Pass necessary arguments to SpeedEstimator constructor
        self.vspeed = SpeedEstimator(names, csv_file, lane_regions)  # Corrected initialization
        
        self.target_x = target_x  # X-coordinate for lane checking (if needed)
        self.view_img = view_img  # Whether to display the image or not
        self.line_thickness = line_thickness  # Thickness of lines used in annotations
        self.csv_file = csv_file  # CSV file to log speeds
        self.boxes = None
        self.trk_ids = None
        self.clss = None
        self.dist_data = {}  # Dictionary to store the speed for each track
    
    def extract_tracks(self, tracks):
        """
        Extracts bounding boxes, class labels, and track IDs from YOLO tracking data.

        Args:
            tracks (YOLO object): YOLO detection object containing bounding boxes, class labels, and IDs.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        self.clss = tracks[0].boxes.cls.cpu().numpy()  # Class labels
        self.trk_ids = tracks[0].boxes.id.cpu().numpy()  # Tracking IDs

 

    def check_and_log_lanes(self, im0, tracks, current_time, skipped_frames=0):
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
        self.annotator = Annotator(self.im0, line_width=self.line_thickness)

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            trk_id = int(trk_id)
            if cls != 4:  # Filtering for Class 4 (Sedan) only
                continue

            track = self.lane_checker.store_track_info(trk_id, box)
            bbox_center = track[-1]
            class_label = self.names[cls] if cls < len(self.names) else 'Unknown'
            
            lane_idx = self.lane_checker.get_persistent_lane_index(trk_id)
            if lane_idx == -1 and self.lane_checker.check_y_coordinate(trk_id, bbox_center):
                lane_idx = self.lane_checker.check_lane(trk_id, bbox_center)
                self.lane_checker.lane_indices[trk_id] = lane_idx  # Store the lane index once determined

                if lane_idx != -1:
                    self.lane_checker.log_lane_entry(trk_id, lane_idx, current_time.strftime('%Y-%m-%d %H:%M:%S'))

                    # **Calculate speed after lane is determined**
                    speed = self.vspeed.calculate_speed(trk_id, track)  # Call on instance `self.vspeed`
                    if speed is not None:
                      self.vspeed.log_speed_to_csv(trk_id, speed)    # Log the speed if calculated

            # Plot the bounding box and tracking path
            self.lane_checker.plot_box_and_track(
                self.im0,
                trk_id,
                box,
                cls,
                track,
                lane_idx,
                self.annotator,
                class_label
            )
            self.vspeed.plot_box_and_track( trk_id, box, cls, track, im0)

        self.im0 = self.annotator.result()
        return self.im0
