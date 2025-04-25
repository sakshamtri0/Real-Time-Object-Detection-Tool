import cv2
import numpy as np
import urllib.request

# URL to capture image
url = 'http://192.168.105.213/cam-hi.jpg'

# Initialize parameters
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# COCO class names (from coco.names)
classNames = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Classification into bio, non-bio, and e-waste
category_map = {
    'bio': ['banana','person','apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake','papers','cardboard'],
    'non-bio': ['bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
                'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                'tennis racket','spoon', 'bowl', 'chair', 'sofa', 'pottedplant', 
                'bed', 'diningtable', 'toilet', 'sink', 'refrigerator', 'book', 'vase', 'scissors', 'teddy bear', 'toothbrush','bottle','wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
    'e-waste': ['tvmonitor', 'laptop', 'mouse', 'remote', 
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'refrigerator', 'clock', 'vase', 'scissors', 
                'hair drier', 'toothbrush']
}

# Load YOLO model
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    found_bio = False
    found_non_bio = False
    found_e_waste = False

    # Loop through YOLO outputs and process detections
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Ensure indices is iterable even if it's a scalar
    if isinstance(indices, tuple) and len(indices) > 0:
        indices = indices[0]

    if indices is not None and len(indices) > 0:
        indices = indices.flatten()

    print(indices)

    # Classify the objects and draw bounding boxes
    for i in indices:
        if i >= len(classIds):
            continue

        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        classId = classIds[i]
        label = classNames[classId]

        # Check if the label belongs to bio, non-bio, or e-waste
        classification = ""
        if label in category_map['bio']:
            classification = "bio"
            found_bio = True
        elif label in category_map['non-bio']:
            classification = "non-bio"
            found_non_bio = True
        elif label in category_map['e-waste']:
            classification = "e-waste"
            found_e_waste = True

        # Draw bounding box and label
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(im, f'{label.upper()} {classification} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return found_bio, found_non_bio, found_e_waste


# Capture video and detect objects
while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    # Prepare image for YOLO input
    blob = cv2.dnn.blobFromImage(im, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    # Find and classify objects
    found_bio, found_non_bio, found_e_waste = findObject(outputs, im)

    # Display the image with the detections
    cv2.imshow('Image', im)
    cv2.waitKey(1)
