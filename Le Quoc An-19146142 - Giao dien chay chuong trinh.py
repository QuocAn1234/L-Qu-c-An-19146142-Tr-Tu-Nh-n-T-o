import tkinter as tk
from tkinter import *
from PIL import ImageTk
from PIL import ImageTk, Image
import sqlite3
from numpy import random
import pyglet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import mediapipe as mp
from keras.preprocessing import image
import imutils
from imutils.video import FPS
from tkinter import messagebox

# set colours
bg_colour = "#3d6466"

# load custom fonts
pyglet.font.add_file("C:/Users/Admin/Downloads/Ubuntu-Bold.ttf")
pyglet.font.add_file("C:/Users/Admin/Downloads/Shanti-Regular.ttf")
######################################
def clear_widgets(frame):
	# select all frame widgets and delete them
	for widget in frame.winfo_children():
		widget.destroy()
##########################################
# initiallize app with basic settings
root = tk.Tk()
root.title("HỆ THỐNG NHẬN DIỆN GÓC QUAY KHUÔN MẶT")
root.eval("tk::PlaceWindow . center")
##########################################
# create a frame widgets
frame1 = tk.Frame(root, width=500, height=600, bg=bg_colour)
frame2 = tk.Frame(root, bg=bg_colour)
###########################################
frame1.tkraise()
	# prevent widgets from modifying the frame
frame1.pack_propagate(False)

	# create logo widget
logo_img = ImageTk.PhotoImage(file="C:/Users/Admin/Downloads/Q.jpg")
logo_widget = tk.Label(frame1, image=logo_img, bg=bg_colour)
logo_widget.image = logo_img
logo_widget.pack()
lbl = Label(root, text="Hệ thống nhận diện góc quay khuôn mặt", fg="blue",font=("Arial Bold", 30)).place(x=50, y=100)
    #Xác định vị trí của label


# place frame widgets in window
for frame in (frame1, frame2):
	frame.grid(row=0, column=0, sticky="nesw")
############    ########
#Thêm một nút nhấn Click Me
def clicked():
    messagebox.showinfo("Please wait!", "Xin vui lòng đợi trong giây lát, bấm OK để tiếp tục")
    ########################################################################################################
    def detect_face_points(image):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("C:/Users/Admin/Downloads/data.dat")
        face_rect = detector(image, 1)
        if len(face_rect) != 1: return []
        dlib_points = predictor(image, face_rect[0])
        face_points = []
        for i in range(68):
             x, y = dlib_points.part(i).x, dlib_points.part(i).y
             face_points.append(np.array([x, y]))
        return face_points
    def compute_features(face_points):
        assert (len(face_points) == 68),messagebox.showerror("ERROR", "Xin vui lòng không quay mặt góc quá lớn")
        face_points = np.array(face_points)
        features = []
        for i in range(68): 
            for j in range(i+1, 68):
                 features.append(np.linalg.norm(face_points[i]-face_points[j]))
        return np.array(features).reshape(1, -1)
    cap = cv2.VideoCapture(0)
    model = load_model('C:/Users/Admin/Desktop/models/modelface.h5')
    while(True):
        success, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.flip(image,1)
        face_points = detect_face_points(image)
        for x, y in face_points:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        features = compute_features(face_points)
        y_pred = model.predict(features)
        roll_pred, pitch_pred, yaw_pred = y_pred[0]
        print(' Z: {:.2f}°'.format(roll_pred))
        print('X: {:.2f}°'.format(pitch_pred))
        print('  Y: {:.2f}°'.format(yaw_pred))
        if yaw_pred < -350:
            text = "Nhin ben trai"
        elif yaw_pred > 350:
            text = "Nhin ben phai"
        elif pitch_pred < -150:
            text = "Nhin xuong duoi"
        elif pitch_pred>150 :
            text = "Nhin len tren"
        elif roll_pred>-150 :
            text = "Nghieng phai"
        elif roll_pred<-260:
            text = "Nghieng trai"
        else:
            text = "Nhin thang"
        cv2.putText(image,"Huong nhin:" ,(100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, text, (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, 'X:'+str(np.round(pitch_pred,2)),(460,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "Y:"+str(np.round(yaw_pred,2)),(460,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "Z:"+str(np.round(roll_pred,2)),(460,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "Goc Quay:",(460,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('GOC XOAY KHUON MAT', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    
       
btn = Button(root, text="Bấm để nhận diện", bg="white",font=("Arial Bold", 20), fg="blue",height= 1, width=50,command=clicked)
#Thiết lập vị trí của nút nhấn có màu nền và màu chữ
btn.grid(row=1, column=0)

# run app
root.mainloop()
	
