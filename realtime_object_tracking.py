import cv2
import torch

# Load YOLOv5 model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def track_objects():
    cap = cv2.VideoCapture(0)  # Use your default webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference on the frame
        results = model(frame)
        # Render results on the frame
        annotated_frame = results.render()[0]
        
        cv2.imshow("Realtime Object Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_objects()
