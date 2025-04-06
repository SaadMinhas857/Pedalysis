import cv2
from pedestrianpsm.LaneChecker import LaneChecker
from ultralytics.utils.plotting import Annotator, colors
class PEDESTRIANESTIMATOR:
    def __init__(self, names, lane_regions, lane_centers, view_img=False, line_thickness=2, csv_file="lane_log.csv"):
        """
        Initializes the SpeedEstimator2 class with specific settings.

        Args:
            names (list): List of names corresponding to detected object classes.
            lane_regions (list): List of tuples defining the polygon regions of each lane.
            lane_centers (list): List of centers for each lane.
            view_img (bool): Flag to determine if the image should be displayed.
            line_thickness (int): Line thickness for annotations.
            csv_file (str): Path to CSV file for logging lane entries and other events.
        """
        self.names = names  # Object class names from YOLO model
        self.lane_checker = LaneChecker(lane_regions, lane_centers, csv_file)  # Initialize LaneChecker with lane regions and centers
        self.view_img = view_img  # Whether to display the image or not
        self.line_thickness = line_thickness  # Thickness of lines used in annotations
        self.csv_file = csv_file  # CSV file to log speeds
        self.boxes = None
        self.trk_ids = None
        self.clss = None

    def extract_tracks(self, tracks):
        """
        Extracts bounding boxes, class labels, and track IDs from YOLO tracking data.

        Args:
            tracks (YOLO object): YOLO detection object containing bounding boxes, class labels, and IDs.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        self.clss = tracks[0].boxes.cls.cpu().numpy()  # Class labels
        self.trk_ids = tracks[0].boxes.id.cpu().numpy()  # Tracking IDs


    def process_frame(self, im0, tracks, current_time):
        """
        Processes a frame, extracting track information and performing lane checks.

        Args:
            im0 (ndarray): The input image frame.
            tracks (list): List of tracked objects in the frame.
            current_time (str): Current timestamp for logging.
        """
        self.im0 = im0  # Save the frame for annotation
        if tracks is None or len(tracks) == 0:
            return self.im0  # If no tracks are detected, return the frame as is
        
        # Extract track information
        self.extract_tracks(tracks)
        self.annotator = Annotator(self.im0, line_width=self.line_thickness)
        
        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            trk_id = int(trk_id)
            if cls != 3:  # Filtering for Class 4 (Sedan) only; adjust if needed
                continue
            track = self.lane_checker.store_track_info(trk_id, box)
            bbox_center = track[-1]
            class_label = self.names[cls] if cls < len(self.names) else 'Unknown'

            # Store tracking info
        

            # Use LaneChecker to handle lane entry/exit and midpoint crossing
            self.lane_checker.check_crossing(trk_id, box, current_time , bbox_center)  # Pass bounding box directly and current_time

            # Plot the bounding box and tracking path
            self.lane_checker.plot_box_and_track(
                self.im0,
                trk_id,
                box,
                cls,
                track,
                self.annotator,
                class_label
            )
            
        
        # Annotate the frame
        self.im0 = self.annotator.result()
        return self.im0
