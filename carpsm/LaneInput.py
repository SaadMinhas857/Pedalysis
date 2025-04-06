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

    def get_lanes(self):
        """
        Returns the list of lane regions.

        Returns:
            list: List of lane regions, each defined by four points.
        """
        return self.lane_regions
