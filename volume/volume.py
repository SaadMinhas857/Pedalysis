from collections import defaultdict
from datetime import datetime, timedelta

class VehicleDetectionWithTracks:
    def __init__(self, tracks, start_time="2025-02-27 08:00:00", frame_rate=30):
        """
        Initialize the VehicleDetectionWithTracks class with pre-detected vehicle tracks.
        :param tracks: List of vehicle tracks where each track is a tuple (frame_number, bounding_box).
                       Example: [(frame_number, (xmin, ymin, xmax, ymax)), ...]
        :param start_time: Start time of the video or detection, used to generate timestamps.
        :param frame_rate: The frame rate of the video (frames per second).
        """
        self.tracks = tracks
        self.start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        self.frame_rate = frame_rate
        self.vehicle_timestamps = []

        # Generate synthetic timestamps based on frame number
        self.extract_timestamps()

    def extract_timestamps(self):
        """
        Generate synthetic timestamps based on the frame number for each detection.
        """
        for frame_number, bbox in self.tracks:
            # Calculate timestamp by adding time based on frame number and frame rate
            seconds_elapsed = frame_number / self.frame_rate
            timestamp = self.start_time + timedelta(seconds=seconds_elapsed)
            self.vehicle_timestamps.append(timestamp)

    def calculate_traffic_flow(self):
        """
        Calculate the number of vehicles passing each hour and 10-minute interval.
        :return: Dictionary of vehicles per hour and per 10-minute interval, and max flow interval.
        """
        # Group vehicles by 10-minute intervals
        interval_vehicles = defaultdict(int)  # Store vehicle count per 10-minute interval
        
        # Start of the first interval
        start_time = min(self.vehicle_timestamps).replace(second=0, microsecond=0)
        
        # Group timestamps into 10-minute intervals
        for ts in self.vehicle_timestamps:
            interval_start = start_time + timedelta(minutes=(ts.minute // 10) * 10)
            interval_vehicles[interval_start] += 1
        
        # Calculate total number of vehicles that passed in the hour
        vehicles_per_hour = defaultdict(int)
        for ts in self.vehicle_timestamps:
            hour_start = ts.replace(minute=0, second=0, microsecond=0)
            vehicles_per_hour[hour_start] += 1
        
        # Determine the 10-minute interval with the greatest flow
        max_flow_interval = max(interval_vehicles, key=interval_vehicles.get)
        max_flow_count = interval_vehicles[max_flow_interval]

        return vehicles_per_hour, interval_vehicles, max_flow_interval, max_flow_count
