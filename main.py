from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter
import os
from time import strftime
from datetime import datetime
from basketballShot import BasketballShot
from double_dribble import DoubleDribbleDetector
from holding_basketball import BallHoldingDetector
from travel_detection import TravelDetection


class BasketballAnalysis :
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        # First Image
        img = Image.open(r"images/image1.jpeg")
        img = img.resize((500, 130), Image.Resampling.LANCZOS)
        self.photoimg = ImageTk.PhotoImage(img)
        f_lbl = Label(self.root, image=self.photoimg)
        f_lbl.place(x=0, y=0, width=500, height=130)

        # Second Image
        img1 = Image.open(r"images/image2.jpg")
        img1 = img1.resize((500, 130), Image.Resampling.LANCZOS)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        f_lbl = Label(self.root, image=self.photoimg1)
        f_lbl.place(x=500, y=0, width=500, height=130)

        # Third Image
        img2 = Image.open(r"images/image3.jpeg")
        img2 = img2.resize((540, 130), Image.Resampling.LANCZOS)
        self.photoimg2 = ImageTk.PhotoImage(img2)
        f_lbl = Label(self.root, image=self.photoimg2)
        f_lbl.place(x=1000, y=0, width=540, height=130)

        # Background Image
        img3 = Image.open(r"images/bg3.jpeg")
        img3 = img3.resize((1530, 710), Image.Resampling.LANCZOS)
        self.photoimg3 = ImageTk.PhotoImage(img3)
        bg_img = Label(self.root, image=self.photoimg3)
        bg_img.place(x=0, y=130, width=1530, height=710)

        # Basketball player analysis
        title_lbl = Label(bg_img, text="Basketball Shot Prediction and Player Analysis Using Computer Vision", font=("times new roman", 28, "bold"), bg="black", fg="orange")
        title_lbl.place(x=0, y=0, width=1530, height=47)

        # Time
        def time():
            string = strftime('%I:%M:%S %p')
            lbl.config(text=string)
            lbl.after(1000, time)

        lbl = Label(bg_img, font=('times new roman', 14, 'bold'), background='violet', foreground='blue')
        lbl.place(x=1400, y=600, width=110, height=50)
        time()

        # Basketball Shot prediction
        img4 = Image.open(r"images/prediction1.jpeg")
        img4 = img4.resize((220, 220), Image.Resampling.LANCZOS)
        self.photoimg4 = ImageTk.PhotoImage(img4)
        b1 = Button(bg_img, image=self.photoimg4, command=self.basketballShot, cursor="hand2")
        b1.place(x=220, y=100, width=220, height=220)
        b1 = Button(bg_img, text="Basketball Shot prediction", command=self.basketballShot, cursor="hand2", font=("times new roman", 15,),
                    bg="darkblue", fg="white")
        b1.place(x=220, y=300, width=220, height=40)
        
        #Double Dribble Detection
        img8=Image.open(r"images/prediction2.jpeg")
        img8=img8.resize((220,220),Image.Resampling.LANCZOS)
        self.photoimg8=ImageTk.PhotoImage(img8)
        
        b1=Button(bg_img,image=self.photoimg8,cursor="hand2",command=self.Double_Dribble_Detection)
        b1.place(x=220,y=380,width=220,height=220)
        
        b1=Button(bg_img,text="Double Dribble Detection",cursor="hand2",command=self.Double_Dribble_Detection,font=("times new roman",15,),bg="darkblue",fg="white")
        b1.place(x=220,y=580,width=220,height=40)


        #Basketball hold Detection
        img7=Image.open(r"images/holding.jpeg")
        img7=img7.resize((220,220),Image.Resampling.LANCZOS)
        self.photoimg7=ImageTk.PhotoImage(img7)
        b1=Button(bg_img,image=self.photoimg7,cursor="hand2",command=self.Basketball_hold_Detection)
        b1.place(x=1100,y=100,width=220,height=220)
        b1=Button(bg_img,text="Basketball hold Detection",cursor="hand2",command=self.Basketball_hold_Detection,font=("times new roman",15,),bg="darkblue",fg="white")
        b1.place(x=1100,y=300,width=220,height=40)
        
        # travel Detection
        img11=Image.open(r"images/travel.jpg")
        img11=img11.resize((220,220),Image.Resampling.LANCZOS)
        self.photoimg11=ImageTk.PhotoImage(img11)
        b1=Button(bg_img,image=self.photoimg11,cursor="hand2",command=self.travel_Detection)
        b1.place(x=1100,y=380,width=220,height=220)
        b1=Button(bg_img,text="Player Travel Detection",cursor="hand2",command=self.travel_Detection,font=("times new roman",15,),bg="darkblue",fg="white")
        b1.place(x=1100,y=580,width=220,height=40)


    # Function Button
    def basketballShot(self):
        basketball_app = BasketballShot()
        if basketball_app.select_video() and basketball_app.select_roi():
            basketball_app.process_video()

    def Double_Dribble_Detection(self):
        self.app = DoubleDribbleDetector()  # Create an instance of the detector
        self.app.run()  # Call its `run` method to start processing
        
    
    def Basketball_hold_Detection(self):
        self.app = BallHoldingDetector()  # Create an instance of the detector
        self.app.run()  # Call its `run` method to start processing
    
    def travel_Detection(self):
        self.app = TravelDetection()  # Create an instance of the detector
        self.app.run()  # Call its `run` method to start processing


if __name__ == "__main__":
    root = Tk()
    obj = BasketballAnalysis(root)
    root.mainloop()
