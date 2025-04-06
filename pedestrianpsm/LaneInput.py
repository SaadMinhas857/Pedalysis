import cv2
import numpy as np

class LaneInput:
    def __init__(self):
        self.lane_regions = []  # List to store lane regions (quadrilaterals)
        self.num_lanes = 0  # Number of lanes to be entered
        self.points = []  # Temporary list to store points for the current lane

    def input_lanes(self, image):
        """
        Opens an OpenCV window to input points for lanes.
        First, input the number of lanes, then select the points interactively for each lane.

        Args:
            image (ndarray): The image where lanes will be drawn.
        """
        print("Please input the number of lanes:")
        self.num_lanes = int(input())  # Input number of lanes
        
        # If the number of lanes is 0, use default regions
        if self.num_lanes == 0:
            print("No lanes inputted. Using default lane regions.")
            self.lane_regions = [((509, 2), (622, 832), (988, 832), (731, 3)), ((733, 4), (987, 832), (1342, 833), (969, 5)), ((969, 5), (1344, 832), (1528, 832), (1233, 2))] 
            self.num_lanes = len(self.lane_regions)  # Set number of lanes to match default regions
            print(f"Default Lane Regions: {self.lane_regions}")
            return

        def draw_lane(event, x, y, flags, param):
            """Handles mouse events to select points for lanes and draw quadrilaterals."""
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                print(f"Point selected: ({x}, {y})")

                # Draw a small circle at each selected point for visual feedback
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select Lane Points", image)

                # If four points (quadrilateral) are selected, draw the lane
                if len(self.points) == 4:
                    # Draw the quadrilateral (lane) on the image
                    pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.imshow("Select Lane Points", image)

                    # Add the quadrilateral as a lane region (four points)
                    self.lane_regions.append(tuple(self.points))
                    print(f"Lane region added: {self.lane_regions[-1]}")

                    # Clear points for the next lane
                    self.points = []

        # Display the image and set up the mouse callback for point selection
        cv2.namedWindow("Select Lane Points")
        cv2.setMouseCallback("Select Lane Points", draw_lane)
        cv2.imshow("Select Lane Points", image)
        cv2.waitKey(0)  # Wait for the user to finish selecting all lanes
        cv2.destroyAllWindows()

    def calculate_polygon_center(self, polygon):
        """Calculates the center (centroid) of a polygon defined by four points."""
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        return (center_x, center_y)

    def get_lanes_and_centers(self):
        """
        Returns the list of lane regions and their centers.

        Returns:
            tuple: (lane_regions, lane_centers)
        """
        lane_centers = [self.calculate_polygon_center(lane) for lane in self.lane_regions]
        print(f"Lane Centers: {lane_centers}")
        return self.lane_regions, lane_centers
