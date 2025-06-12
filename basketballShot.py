from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math
from scipy.spatial import distance


class BasketballShot:
    def __init__(self):
        self.cap = None
        self.hsvVals = {'hmin': 0, 'smin': 76, 'vmin': 0, 'hmax': 23, 'smax': 255, 'vmax': 255}
        self.myColorFinder = ColorFinder(False)
        self.posListX = []
        self.posListY = []
        self.listX = [item for item in range(0, 1300)]
        self.start = True
        self.prediction = False
        self.basket_y = 200
        self.basket_x_min = 240
        self.basket_x_max = 300
        self.roi = None
        self.coff = None
        self.pred_text = ""
        self.pred_color = (0, 0, 200)
        self.frame_rate = None
        self.time_interval = None
        self.conversion_factor = 0.01

    def calculate_speed(self, x1, y1, x2, y2, time_interval):
        dist = distance.euclidean((x1, y1), (x2, y2))
        speed = dist / time_interval * self.conversion_factor
        return speed

    def calculate_angle(self, x1, y1, x2, y2):
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if not file_path:
            return False
        self.cap = cv2.VideoCapture(file_path)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_interval = 1 / self.frame_rate
        return True

    def select_roi(self):
        success, img = self.cap.read()
        if not success:
            print("Failed to read the video")
            return False
        self.roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning
        return True

    def process_video(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break

            imgRoi = img[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])]
            imgResult = imgRoi.copy()
            imgBall, mask = self.myColorFinder.update(imgRoi, self.hsvVals)
            imgCon, contours = cvzone.findContours(img, mask, 200)

            if contours:
                self.posListX.append(contours[0]['center'][0])
                self.posListY.append(contours[0]['center'][1])

            if len(self.posListX) == 10 and self.start:
                self.start = False
                self.coff = np.polyfit(self.posListX, self.posListY, 2)
                a, b, c = self.coff
                c = c - self.basket_y
                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0:
                    x1 = int((-b - math.sqrt(discriminant)) / (2 * a))
                    x2 = int((-b + math.sqrt(discriminant)) / (2 * a))
                    self.prediction = (self.basket_x_min < x1 < self.basket_x_max) or (self.basket_x_min < x2 < self.basket_x_max)
                else:
                    self.prediction = False

                if self.prediction:
                    self.pred_text = "Basket"
                    self.pred_color = (0, 200, 0)
                else:
                    self.pred_text = "No Basket"
                    self.pred_color = (0, 0, 200)

            # Draw past positions and prediction
            for i, (posX, posY) in enumerate(zip(self.posListX, self.posListY)):
                pos = (posX, posY)
                cv2.circle(imgCon, pos, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(imgResult, pos, 10, (0, 255, 0), cv2.FILLED)
                if i > 0:
                    cv2.line(imgCon, (self.posListX[i - 1], self.posListY[i - 1]), pos, (0, 255, 0), 2)
                    cv2.line(imgResult, (self.posListX[i - 1], self.posListY[i - 1]), pos, (0, 255, 0), 2)

            # Calculate speed, angle, and distance
            if len(self.posListX) > 1:  # Only start calculating once there are at least two positions
                x1, y1 = self.posListX[-2], self.posListY[-2]  # Previous position
                x2, y2 = self.posListX[-1], self.posListY[-1]  # Current position
                speed = self.calculate_speed(x1, y1, x2, y2, self.time_interval)
                angle = self.calculate_angle(x1, y1, x2, y2)
                distance_traveled = distance.euclidean((x1, y1), (x2, y2))

                # Display speed, angle, and distance on the image with larger font
                cvzone.putTextRect(img, f"Speed: {speed:.2f} m/s", (900, 50), colorR=(0, 255, 0), scale=2, thickness=4, offset=6)
                cvzone.putTextRect(img, f"Angle: {angle:.2f} degrees", (900, 100), colorR=(0, 255, 0), scale=2, thickness=4, offset=6)
                cvzone.putTextRect(img, f"Distance: {distance_traveled:.2f} pixels", (900, 150), colorR=(0, 255, 0), scale=2, thickness=4, offset=6)


            if self.coff is not None:
                for x in self.listX:
                    y = int(self.coff[0] * x ** 2 + self.coff[1] * x + self.coff[2])
                    cv2.circle(imgResult, (x, y), 2, (255, 0, 255), cv2.FILLED)

                cvzone.putTextRect(imgResult, self.pred_text, (50, 200), colorR=self.pred_color, scale=5, thickness=10, offset=20)

            # Overlay ROI back to the original frame
            img[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])] = cv2.resize(imgResult, (self.roi[2], self.roi[3]))
            img = cv2.resize(img, (0, 0), None, 0.7, 0.7)

            cv2.imshow("Full Video with Detection", img)
            key = cv2.waitKey(100)
            if key == 13:  # Exit on pressing 'ESC'
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the main window
    app = BasketballShot()

    if app.select_video() and app.select_roi():
        app.process_video()
