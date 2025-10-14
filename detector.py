from ultralytics import YOLO
import cv2
import time

class PersonDetector:
    def __init__(self, source=0):
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(source)
        self.fps = 0

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            start_time = time.time()
            results = self.model(frame)
            annotated_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    if label == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            end_time = time.time()
            self.fps = 1 / (end_time - start_time)
            cv2.putText(annotated_frame, f"FPS: {int(self.fps)}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def __del__(self):
        self.cap.release()
