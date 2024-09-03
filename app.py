from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
MODEL_PATH = 'yolov8/best-4.pt'
model = YOLO(MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.4  # Confidence threshold set to 40%

def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = model(frame)  # Get predictions on the frame
        
        # Filter out detections below the confidence threshold
        filtered_results = []
        for result in results[0].boxes:
            if result.conf >= CONFIDENCE_THRESHOLD:
                filtered_results.append(result)
        
        # If there are any filtered results, annotate the frame
        if filtered_results:
            annotated_frame = results[0].plot(boxes=filtered_results)  # Annotate the frame with filtered detections
        else:
            annotated_frame = frame  # If no results pass the threshold, show the original frame

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Frame response for streaming

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
