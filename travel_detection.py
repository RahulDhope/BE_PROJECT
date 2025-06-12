import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import deque
import os
import time

class TravelDetection:
    def __init__(self):
        # Initialize YOLO models
        self.ball_model = YOLO("basketballModel.pt")
        self.pose_model = YOLO("yolo11n-pose.pt")

        # Initialize counters and thresholds
        self.dribble_count = 0
        self.step_count = 0
        self.total_dribble_count = 0
        self.total_step_count = 0

        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.prev_delta_y = None
        self.ball_not_detected_frames = 0
        self.max_ball_not_detected_frames = 20
        self.dribble_threshold = 18
        self.step_threshold = 5
        self.min_wait_frames = 7
        self.wait_frames = 0
        self.travel_detected = False
        self.travel_timestamp = None

        # Frame buffer
        self.frame_buffer = deque(maxlen=30)
        self.save_frames = 60
        self.frame_save_counter = 0
        self.saving = False
        self.out = None

        # Body part indices
        self.body_index = {"left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}

        # Create directory for travel footage
        if not os.path.exists("travel_footage"):
            os.makedirs("travel_footage")

    def select_video_file(self):
        Tk().withdraw()
        file_path = askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        return file_path

    def process_frame(self, frame):
        # Perform ball and pose detection and handle logic
        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)
        ball_detected = False

        for results in ball_results_list:
            for bbox in results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                if self.prev_y_center is not None:
                    delta_y = y_center - self.prev_y_center
                    if (
                        self.prev_delta_y is not None
                        and self.prev_delta_y > self.dribble_threshold
                        and delta_y < -self.dribble_threshold
                    ):
                        self.dribble_count += 1
                        self.total_dribble_count += 1

                    self.prev_delta_y = delta_y

                self.prev_x_center = x_center
                self.prev_y_center = y_center
                ball_detected = True
                self.ball_not_detected_frames = 0

            annotated_frame = results.plot()

        if not ball_detected:
            self.ball_not_detected_frames += 1
            if self.ball_not_detected_frames >= self.max_ball_not_detected_frames:
                self.step_count = 0

        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        keypoints = pose_results[0].keypoints.xy.numpy()
        rounded_results = np.round(keypoints, 1)

        try:
            left_ankle = rounded_results[0][self.body_index["left_ankle"]]
            right_ankle = rounded_results[0][self.body_index["right_ankle"]]

            if self.prev_left_ankle_y is not None and self.wait_frames == 0:
                left_diff = abs(left_ankle[1] - self.prev_left_ankle_y)
                right_diff = abs(right_ankle[1] - self.prev_right_ankle_y)
                if max(left_diff, right_diff) > self.step_threshold:
                    self.step_count += 1
                    self.total_step_count += 1

                self.wait_frames = self.min_wait_frames

            self.prev_left_ankle_y = left_ankle[1]
            self.prev_right_ankle_y = right_ankle[1]
            if self.wait_frames > 0:
                self.wait_frames -= 1

        except Exception:
            print("No human detected.")

        pose_annotated_frame = pose_results[0].plot()
        combined_frame = cv2.addWeighted(annotated_frame, 0.6, pose_annotated_frame, 0.4, 0)
        return combined_frame

    def run(self):
        video_file = self.select_video_file()
        if not video_file:
            print("No video file selected. Exiting...")
            return

        cap = cv2.VideoCapture(video_file)
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                self.frame_buffer.append(frame)
                combined_frame = self.process_frame(frame)
                cv2.imshow("Travel Detection", combined_frame)

                # Allow exit on 'q' key press or window close
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q") or cv2.getWindowProperty("Travel Detection", cv2.WND_PROP_VISIBLE) < 1:
                    print("Processing stopped by user.")
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Release resources and destroy windows
            cap.release()
            cv2.destroyAllWindows()
            print("Resources released. Program exited.")




if __name__ == "__main__":
    app = TravelDetection()
    app.run()
