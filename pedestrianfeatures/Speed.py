import csv
from datetime import datetime, timedelta
import numpy as np

class PedestrianSpeedCalculator:
    def __init__(self, csv_file="pedestrian_speed_log.csv", pixel_to_meter=0.01):
        """
        Initializes the PedestrianSpeedCalculator with the required settings.

        Args:
            csv_file (str): File path for logging pedestrian speed data.
            pixel_to_meter (float): Conversion factor for pixel displacement to meters.
        """
        self.csv_file = csv_file
        self.pixel_to_meter = pixel_to_meter
        self.init_csv()

    def init_csv(self):
        """Initializes the CSV file with headers for pedestrian speed data."""
        header = ["Track ID", "Start Time", "End Time", "Start Position (x, y)", "End Position (x, y)", "Distance (m)", "Speed (m/s)"]

        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def log_speed(self, trk_id, start_time, end_time, start_pos, end_pos, distance, speed):
        """
        Logs the calculated speed data into the CSV file.

        Args:
            trk_id (int): The tracking ID of the object.
            start_time (str): The start time when tracking began.
            end_time (str): The end time when tracking stopped.
            start_pos (tuple): The start position (x, y) of the pedestrian.
            end_pos (tuple): The end position (x, y) of the pedestrian.
            distance (float): The total distance traveled by the pedestrian (in meters).
            speed (float): The calculated speed (in meters per second).
        """
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([trk_id, start_time, end_time, start_pos, end_pos, distance, speed])

    def calculate_distance(self, start_pos, end_pos):
        """
        Calculates the Euclidean distance between the start and end positions.

        Args:
            start_pos (tuple): The start position (x, y) of the pedestrian.
            end_pos (tuple): The end position (x, y) of the pedestrian.
        
        Returns:
            float: The distance between the start and end positions in pixels.
        """
        return np.linalg.norm(np.array(end_pos) - np.array(start_pos))

    def calculate_speed(self, start_pos, end_pos, start_time, end_time):
        """
        Calculates the pedestrian speed based on the start and end positions and times.

        Args:
            start_pos (tuple): The start position (x, y) of the pedestrian.
            end_pos (tuple): The end position (x, y) of the pedestrian.
            start_time (str): The start time when tracking began.
            end_time (str): The end time when tracking stopped.

        Returns:
            float: The calculated speed in meters per second (m/s).
        """
        # Calculate distance in pixels
        distance_pixels = self.calculate_distance(start_pos, end_pos)
        
        # Convert distance to meters
        distance_meters = distance_pixels * self.pixel_to_meter
        
        # Convert times to datetime objects
        start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_time_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        
        # Calculate the time difference in seconds
        time_diff_seconds = (end_time_dt - start_time_dt).total_seconds()
        
        # Calculate speed (m/s)
        speed = distance_meters / time_diff_seconds if time_diff_seconds > 0 else 0

        return distance_meters, speed

    def process_data(self, trk_id, start_time, end_time, start_pos, end_pos):
        """
        Processes the data, calculates the speed, and logs it to the CSV file.

        Args:
            trk_id (int): The tracking ID of the object.
            start_time (str): The start time when tracking began.
            end_time (str): The end time when tracking stopped.
            start_pos (tuple): The start position (x, y) of the pedestrian.
            end_pos (tuple): The end position (x, y) of the pedestrian.
        """
        # Calculate distance and speed
        distance, speed = self.calculate_speed(start_pos, end_pos, start_time, end_time)
        
        # Log the data to the CSV file
        self.log_speed(trk_id, start_time, end_time, start_pos, end_pos, distance, speed)

        return distance, speed

# Example usage:
# Initialize the PedestrianSpeedCalculator
speed_calculator = PedestrianSpeedCalculator(csv_file="pedestrian_speeds.csv")

# Example data (start and end positions, timestamps)
trk_id = 1  # Tracking ID
start_time = "2025-03-06 12:30:00"  # Start time
end_time = "2025-03-06 12:30:10"  # End time
start_pos = (50, 50)  # Start position (x, y)
end_pos = (100, 100)  # End position (x, y)

# Process the data for speed calculation
distance, speed = speed_calculator.process_data(trk_id, start_time, end_time, start_pos, end_pos)

# Output the results
print(f"Pedestrian {trk_id} traveled {distance:.2f} meters at {speed:.2f} m/s.")
