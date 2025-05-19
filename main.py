import cv2
import numpy as np

from collections import deque

STOP_THRESHOLD = 0.05
CANNY_LOW = 50
CANNY_HIGH = 150

net = cv2.dnn.readNetFromCaffe("models\MobileNetSSD_deploy.prototxt.txt", "models\MobileNetSSD_deploy.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0) # capture from webcam (0)

ratio_history = deque(maxlen = 10) # to implement rolling average over last 10 frames
left_zone_hist = deque(maxlen = 10)
center_zone_hist = deque(maxlen = 10)
right_zone_hist = deque(maxlen = 10)

def detect_objects(frame, net, CLASSES, confidence_threshold=0.5):

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{CLASSES[idx]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def highlight_direction_zone(frame, x1, y1, x2, y2, direction_index):

    overlay = frame.copy()
    zone_width = (x2 - x1) // 3

    # Compute zone bounds
    zone_bounds = [
        (x1, x1 + zone_width),
        (x1 + zone_width, x1 + 2 * zone_width),
        (x1 + 2 * zone_width, x2)
    ]
    zone_start, zone_end = zone_bounds[direction_index]

    # Draw translucent overlay on selected zone
    cv2.rectangle(overlay, (zone_start, y1), (zone_end, y2), (0, 255, 0), -1)
    alpha = 0.25
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

while True:

    ret, frame = cap.read()

    if not ret:
        print('Webcam read failed')
        break
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
    height, width = frame.shape[:2] # the third value is channel
    x_inset = int(width  * 0.20)    # 25% from the left and right
    y_inset = int(height * 0.20)    # 25% from the top and bottom

    x1, x2 = x_inset, width - x_inset
    y1, y2 = y_inset, height - y_inset
    
    roi = frame[y1:y2, x1:x2] # region of interest

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # convert to single channel
    edges = cv2.Canny(gray,CANNY_LOW, CANNY_HIGH) # canny edges

    # steering logic
    edges_height, edges_width = edges.shape
    zone_width = edges_width // 3

    left_zone   = edges[:, 0: zone_width]
    center_zone = edges[:, zone_width: 2*zone_width]
    right_zone  = edges[:, 2*zone_width:]

    left_ratio   = np.count_nonzero(left_zone)   / left_zone.size
    center_ratio = np.count_nonzero(center_zone) / center_zone.size
    right_ratio  = np.count_nonzero(right_zone)  / right_zone.size

    # add moving average logic
    left_zone_hist.append(left_ratio)
    center_zone_hist.append(center_ratio)
    right_zone_hist.append(right_ratio)

    avg_left_ratio   = sum(left_zone_hist)   / len(left_zone_hist)
    avg_center_ratio = sum(center_zone_hist) / len(center_zone_hist)
    avg_right_ratio  = sum(right_zone_hist)  / len(right_zone_hist)

    # steer towards least cluttered zone
    zone_ratios = [avg_left_ratio, avg_center_ratio, avg_right_ratio]
    min_index = zone_ratios.index(min(zone_ratios))

    print(f"Zones (L, C, R): {avg_left_ratio:.3f}, {avg_center_ratio:.3f}, {avg_right_ratio:.3f}")

    if min_index == 0:
        direction = "LEFT"
    elif min_index == 1:
        direction = "FORWARD"
    else:
        direction = "RIGHT"
    
    # to detect the obstacles we use edge density
    edge_density = np.sum(edges) / 255
    total_pixels = edges.shape[0] * edges.shape[1]
    ratio = edge_density / total_pixels

    # implement rolling average
    ratio_history.append(ratio)
    avg_ratio = sum(ratio_history) / len(ratio_history)


    if avg_ratio < STOP_THRESHOLD:
        cmd = "STOP"
    else:
        cmd = direction


    # display text over frame window to indicate real time decision
    overlay_text = f"{cmd} | Ratio: {avg_ratio:.3f}"

    cv2.imshow("Edges", edges)
    
    # enhancement for debugging
    cv2.putText(frame, overlay_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2) 
    
    # draw rectagle over ROI
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

    first_div = x1 + zone_width
    second_div = x1 + 2 * zone_width
    cv2.line(frame, (first_div, y1), (first_div, y2), (255, 0, 0), 2)
    cv2.line(frame, (second_div, y1), (second_div, y2), (255, 0, 0), 2)
    if cmd != "STOP": highlight_direction_zone(frame, x1, y1, x2, y2, min_index)

    detect_objects(frame, net, CLASSES)

    cv2.imshow('Webcam', frame)

cap.release()
cv2.destroyAllWindows()