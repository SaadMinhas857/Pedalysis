import numpy as np
import cv2
import csv
import os
import portalocker
from ultralytics.utils.plotting import Annotator, colors

class SpeedEstimator:
    
    def __init__(self, names, csv_file, lane_regions):
        # Initialization logic here
        self.names = names
        self.csv_file = csv_file
        self.lane_regions = lane_regions
        # Additional setup for SpeedEstimator
        
        self.dist_data = {}  # Store distance data for speed calculation
        self.tracks = {}  # Store tracking data
        self.csv_file = csv_file  # CSV file to log speeds
        self.line_thickness = 2  # Thickness of lines used in annotations
    
    def store_track_info(self, trk_id, box):
        """ Stores or updates tracking history for each object based on track ID. """
        if trk_id not in self.tracks:
            self.tracks[trk_id] = []
        self.tracks[trk_id].append(box)  # Append new box to the track's history
        return self.tracks[trk_id]

    def calculate_speed(self, trk_id, track):
        """ Calculates the speed of an object based on tracking history. """
        if len(track) < 2:
            return  # Need at least two points to calculate speed

        prev_point = np.array(track[-2])
        curr_point = np.array(track[-1])
        pixel_distance = np.linalg.norm(curr_point[:2] - prev_point[:2])
        speed_mps = pixel_distance / 0.25  # Example: fixed time interval, adjust as necessary
        speed_kmh = speed_mps * 3.6  # Convert m/s to km/h
        self.dist_data[trk_id] = speed_kmh

        # Log the speed to the CSV file
        self.log_speed_to_csv(trk_id, speed_kmh)

    def log_speed_to_csv(self, track_id, speed):
     """ Logs or updates the speed in the CSV file for the given track ID with file locking. """
     file_exists = os.path.isfile(self.csv_file)

     updated_rows = []
     id_found = False

     # Open the CSV file with locking
     with open(self.csv_file, mode='r+' if file_exists else 'w+', newline='') as file:
        portalocker.lock(file, portalocker.LOCK_EX)  # Acquire an exclusive lock

        reader = csv.reader(file)
        updated_rows = list(reader)

        # Check if track_id exists, and if so, update its speed in the 4th column
        for row in updated_rows:
            if row[0] == str(track_id):
                # Ensure the row has at least 4 columns
                if len(row) < 4:
                    row.extend([''] * (4 - len(row)))  # Add missing columns if necessary

                row[3] = f"{speed:.2f}"  # Update the speed in the 4th column
                id_found = True
                break

        # If the ID was not found, append a new row with the track ID and speed
        if not id_found:
            new_row = [str(track_id), '', '', f"{speed:.2f}"]  # Ensure at least 4 columns
            updated_rows.append(new_row)

        # Move the file pointer back to the start of the file
        file.seek(0)
        file.truncate()  # Clear the contents before rewriting

        # Write the updated rows back to the CSV file
        writer = csv.writer(file)
        writer.writerows(updated_rows)

        portalocker.unlock(file)  # Release the lock

    def plot_box_and_track(self, trk_id, box, cls, track, im0):
        """
        Plots the bounding box and tracking path of the object.

        Args:
            track_id (int): Object track id.
            box (list): Object bounding box data.
            cls (str): Object class name.
            track (list): Tracking history for drawing tracks path.
            im0 (ndarray): The current video frame.
        """
        # Display the speed if calculated, otherwise show the class name
        speed_label = f"{int(self.dist_data[trk_id])} km/h" if trk_id in self.dist_data else self.names[int(cls)]
        
        # Color the bounding box according to the track_id
        bbox_color = colors(int(trk_id)) if trk_id in self.dist_data else (255, 0, 255)

        annotator = Annotator(im0, line_width=self.line_thickness)
        
        # Annotate the bounding box with the speed or class name
        annotator.box_label(box, speed_label, bbox_color)

        # Draw the tracking path of the object
        if len(track) > 1:
            points = np.array([p[:2] for p in track], np.int32)
            cv2.polylines(im0, [points], isClosed=False, color=(0, 255, 0), thickness=1)
            cv2.circle(im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)

        return annotator.result()
