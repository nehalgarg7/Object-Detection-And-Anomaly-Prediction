import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest


def extract_features(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (640, 480))

    flattened = resized.flatten()
    return flattened


def load_frames(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.tif'):
            frame_path = os.path.join(directory, filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)
    return frames


def detect_anomalies(frames):

    features = [extract_features(frame) for frame in frames]

    X = np.array(features)
    
    model = IsolationForest(contamination=0.1)
    model.fit(X)
    
    anomaly_scores = model.decision_function(X)
    
    return anomaly_scores


# Function to draw a red rectangle around anomaly pixels
def draw_rectangle(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)



frames = load_frames('UCSDped1/Test/Test001')

anomaly_scores = detect_anomalies(frames)



for frame, score in zip(frames, anomaly_scores):

    score_rescaled = int(255 * (score - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores)))
    

    color = (255 - score_rescaled, 0, score_rescaled)

    resized_frame = cv2.resize(frame, (640, 480))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# Threshold the grayscale frame to create a binary mask of anomaly pixels
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  
    draw_rectangle(frame, mask)


    cv2.putText(frame, f"Anomaly Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Video Frame", frame)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

