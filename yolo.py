import cv2
from ultralytics import YOLO
from time import monotonic

model = YOLO('yolov9c.pt')
# model.to("cuda")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        print("something went wrong ")
        break
    if success:
        # Run YOLOv9 tracking on the frame, persisting tracks between frames
        start = monotonic()
        results = model.track(frame, persist=True, verbose=False)
        print(f"Processed frame in {(monotonic() - start)*1000:.0f} ms", end="\r")
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv9 Tracking", annotated_frame)
        cv2.waitKey(1)