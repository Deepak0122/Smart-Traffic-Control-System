import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, Text, Scrollbar, END, Label, Button
import cv2
import numpy as np
import os
import tensorflow as tf
from traffic_simulation import Simulation
from yolo_traffic import runYolo

# Disable oneDNN optimizations (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.disable_eager_execution()  # Still needed for older models

# GUI Window
main = tk.Tk()
main.title("Smart Control of Traffic Light Using Artificial Intelligence")
main.geometry("1300x1200")
main.config(bg='snow3')

filename = ""

# Load MobileNetSSD model
net = cv2.dnn.readNetFromCaffe(
    "yolo-coco/MobileNetSSD_deploy.prototxt.txt",
    "yolo-coco/MobileNetSSD_deploy.caffemodel"
)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def yoloTrafficDetection():
    global filename
    filename = filedialog.askopenfilename(initialdir="Videos")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, f"{filename} loaded\n")
    runYolo(filename)

def runSimulation():
    sim = Simulation()
    sim.runSimulation()

def ssdDetection(image_np):
    count = 0
    (h, w) = image_np.shape[:2]
    ssd = tf.Graph()

    with ssd.as_default():
        od_graphDef = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile('yolo-coco/frozen_inference_graph.pb', 'rb') as file:
            serializedGraph = file.read()
            od_graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(od_graphDef, name='')

    with tf.compat.v1.Session(graph=ssd) as sess:
        blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")

                if CLASSES[idx] in {"bicycle", "bus", "car"}:
                    count += 1
                    label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                    cv2.rectangle(image_np, (startX, startY), (endX, endY), COLORS[idx], 2)
                    cv2.putText(image_np, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image_np, f"Detected Count : {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 0, 0), 2, cv2.LINE_AA)
    return image_np

def extensionSingleShot():
    global filename
    filename = filedialog.askopenfilename(initialdir="Videos")
    pathlabel.config(text=filename)
    video = cv2.VideoCapture(filename)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = ssdDetection(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# GUI Layout
font = ('times', 16, 'bold')
Label(main, text='Smart Control of Traffic Light Using Artificial Intelligence',
      bg='light cyan', fg='pale violet red', font=font, height=3, width=120).place(x=0, y=5)

font1 = ('times', 14, 'bold')
Button(main, text="Run Traffic Simulation", command=runSimulation, font=font1).place(x=50, y=100)

pathlabel = Label(main, bg='light cyan', fg='pale violet red', font=font1)
pathlabel.place(x=460, y=100)

Button(main, text="Run Extension Yolo Traffic Detection & Counting", command=yoloTrafficDetection, font=font1).place(x=460, y=150)
Button(main, text="Run Existing Single Shot Traffic Detection", command=extensionSingleShot, font=font1).place(x=50, y=150)

font2 = ('times', 12, 'bold')
text = Text(main, height=20, width=150, font=font2)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=250)

main.mainloop()
