from tkinter import Tk, filedialog
from ultralytics import YOLO
import cv2
import numpy as np
import time

class BallHoldingDetector:
    def __init__(self):
        # Load the YOLO models for pose estimation and ball detection
        self.pose_model = YOLO("yolo11n-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")

        # Initialize the video capture with a file selected by the user
        video_path = self.select_video_file()
        if video_path is None:
            print("No video selected. Exiting...")
            exit()
        self.cap = cv2.VideoCapture(video_path)

        # Define the body part indices. Switch left and right to account for the mirrored image.
        self.body_index = {
            "left_wrist": 10,  # switched
            "right_wrist": 9,  # switched
        }

        # Initialize variables to store the hold start time and the hold flag
        self.hold_start_time = None
        self.is_holding = False

        # Define the holding duration (in seconds)
        self.hold_duration = 0.85

        # Threshold for the distance to be considered as holding
        self.hold_threshold = 300

    def select_video_file(self):
        """Opens a file dialog to select a video file."""
        root = Tk()
        root.withdraw()  # Hide the Tkinter main window
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")),
        )
        return file_path

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                pose_annotated_frame, ball_detected = self.process_frame(frame)

                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", pose_annotated_frame)

                # Exit on 'q' press or window close
                if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("YOLOv8 Inference", cv2.WND_PROP_VISIBLE) < 1:
                    print("Processing stopped by user.")
                    break
            else:
                print("End of video or failed to read frame.")
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()


    def process_frame(self, frame):
        # Perform pose estimation on the frame
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()
        # Access the keypoint coordinates (xy)
        keypoints = pose_results[0].keypoints.xy.numpy()
        # Round the flattened keypoints
        rounded_results = np.round(keypoints, 1)

        try:
            # Get the keypoints for the body parts
            left_wrist = rounded_results[0][self.body_index["left_wrist"]]
            right_wrist = rounded_results[0][self.body_index["right_wrist"]]
        except:
            print("No human detected.")
            return pose_annotated_frame, False

        # Perform ball detection on the frame
        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)

        # Set the ball detection flag to False before the detection
        ball_detected = False

        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2

                print(
                    f"Ball coordinates: (x={ball_x_center:.2f}, y={ball_y_center:.2f})"
                )

                # Update the ball detection flag to True when the ball is detected
                ball_detected = True

                # Calculate distances between the ball and the wrists
                left_distance = np.hypot(
                    ball_x_center - left_wrist[0], ball_y_center - left_wrist[1]
                )
                right_distance = np.hypot(
                    ball_x_center - right_wrist[0], ball_y_center - right_wrist[1]
                )

                # Check if the ball is being held
                self.check_holding(left_distance, right_distance)

                # Annotate ball detection on the pose estimation frame
                cv2.rectangle(
                    pose_annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    pose_annotated_frame,
                    f"Ball: ({ball_x_center:.2f}, {ball_y_center:.2f})",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Apply a blue tint if the ball is being held
                if self.is_holding:
                    blue_tint = np.full_like(
                        pose_annotated_frame, (255, 0, 0), dtype=np.uint8
                    )
                    pose_annotated_frame = cv2.addWeighted(
                        pose_annotated_frame, 0.7, blue_tint, 0.3, 0
                    )

        # If the ball is not detected in the frame, reset the timer and the holding flag
        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False

        return pose_annotated_frame, ball_detected

    def check_holding(self, left_distance, right_distance):
        if min(left_distance, right_distance) < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif (
                time.time() - self.hold_start_time > self.hold_duration
                and not self.is_holding
            ):
                print("The ball is being held.")
                self.is_holding = True
        else:
            self.hold_start_time = None
            self.is_holding = False


if __name__ == "__main__":
    app = BallHoldingDetector()
    app.run()
