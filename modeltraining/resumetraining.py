from ultralytics import YOLO

model = YOLO("path/to/last.pt")

results = model.train(resume = True)