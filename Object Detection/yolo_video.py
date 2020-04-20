# import packages
import numpy as np
import argparse
import time
import cv2
import os
import imutils


# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path for output video")
ap.add_argument("-y", "--yolo", required=True, help="base path to the Yolo directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each label in the class labels
np.random.seed(54)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# define path to Yolo weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our Yolo object detector trained on the COCO dataset (80 classes)
print("[INFO] ... loading Yolo model from disk")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine the output layer names that we need from Yolo
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file and set frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None  # initialize our video writer
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] ... {} total frames in video".format(total))

except:
    print("[INFO] ... could not determine # of frames in the video")
    print("[INFO] ... no approx completion time can be provided")
    total = -1

# Now start processing video frames one-by-one
# loop over frames from the video file stream
while True:
    # read the next frame from the video file
    (grabbed, frame) = vs.read()
    # if a frame was not grabbed, we are at the end of the video
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # perform a forward pass of Yolo using the current frame as the input
    # construct a blob from the input frame and then perform a forward pass
    # of the Yolo object detector giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # define output visualization lists
    boxes, confidences, classIDs = [], [], []

    # Populate the visualization lists from our Yolo forward pass

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class ID and confidence (i.e. probability score) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected probability is
            # greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])  # scales bounding box for overlay on image
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y) coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our output lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    # Note: Yolo does not apply non-maxima suppression for us, so we need to explicitly apply it
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # applying non-maxima suppression suppresses overlapping bounding boxes, keeping only the most
    # confident ones

    # Draw the bounding boxes and class text on the frame
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer - only initialized on the first pass of the loop
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # some info for processing a single frame
        if total > 0:
            elap = end - start
            print("[INFO] ... single frame took {: .4f} seconds".format(elap))
            print("[INFO] ... estimated time to finish: {: .4f} seconds".format(elap * total))

    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] ... cleaning up!")
writer.release()
vs.release()

