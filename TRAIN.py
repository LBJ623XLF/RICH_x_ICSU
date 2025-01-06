from ultralytics import YOLO
import cv2
import numpy as np
from roboflow import Roboflow

rf = Roboflow(api_key="7idEzKSCpPKLTJwlUp3j")
project = rf.workspace("yolov11-midrp").project("hoi-gmd2i")
version = project.version(5)
dataset = version.download("yolov8")

rf = Roboflow(api_key="7idEzKSCpPKLTJwlUp3j")
project = rf.workspace("test-vongh").project("awkward-posture-of-human")
version = project.version(3)
dataset = version.download("yolov8")

model = YOLO('yolov8n-pose.pt')
model = YOLO('yolov8n.pt')


results = model.train(data="/home/idc/ultralytics/HOI-3/data.yaml", epochs=200, imgsz=640)
results = model.train(data="/home/idc/ultralytics/Awkward-posture-of-human-3/data.yaml", epochs=300, imgsz=640)

# model = YOLO('/home/idc/ultralytics/runs/pose/train/weights/best.pt')
# metrics = model.val(data="/home/idc/ultralytics/Awkward-posture-of-human-3/data.yaml")


