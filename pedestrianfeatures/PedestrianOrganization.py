import cv2
from pedestrianfeatures.Speed import PedestrianSpeedCalculator  # Assuming this is the path
from ultralytics.utils.plotting import Annotator, colors

class PedestrianSpeedEstimator:
    def __init__(self, names, view_img=False, line_thickness=2, csv_file="pedestrian_speed_log.csv"):
        """
        Initializes the PedestrianSpeedEstimator class with specific settings.

        Args:
            names (list): List of names corresponding to detected object classes.
            view_img (bool): Flag to determine if the image should be displayed.
            line_thickness (int): Line thickness for annotations.
            csv_file (str): Path to CSV file for logging pedestrian speeds.
        """
        self.names = names  # Object class names from YOLO model
        self.speed_calculator = PedestrianSpeedCalculator(csv_file)  # Initialize the SpeedCalculator for logging speeds
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
        Processes a frame, extracting track information and calculating pedestrian speed.

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
            if cls != 3:  # Filtering for Class 3 (person) only; adjust if needed
                continue
            track = [self.calculate_bbox_center(box)]  # Track contains only the center of the bounding box
            bbox_center = track[-1]
            class_label = self.names[cls] if cls < len(self.names) else 'Unknown'

            # Calculate pedestrian speed (if enough data is available)
            start_pos = track[0]  # Assuming start position is the first tracked position
            end_pos = bbox_center   # End position is the current bounding box center
            if len(track) > 1:  # Ensure there is enough data for speed calculation
                distance, speed = self.speed_calculator.process_data(trk_id, current_time, current_time, start_pos, end_pos)
                label = f'{trk_id}: {class_label}, Speed: {speed:.2f} m/s'
            else:
                label = f'{trk_id}: {class_label}'

            # Annotate the frame with speed and other information
            self.annotator.box_label(box, label, color=colors(cls, True))
        
        # Annotate the frame
        self.im0 = self.annotator.result()
        return self.im0