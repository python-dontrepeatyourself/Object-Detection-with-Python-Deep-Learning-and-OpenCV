import numpy as np
import cv2
import datetime

video_cap = cv2.VideoCapture("examples/video1.mp4")

# grab the width and the height of the video stream
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))
# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# path to the weights and model files
weights = "ssd_mobilenet/frozen_inference_graph.pb"
model = "ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# load the MobileNet SSD model trained  on the COCO dataset
net = cv2.dnn.readNetFromTensorflow(weights, model)

# load the class labels the model was trained on
class_names = []
with open("ssd_mobilenet/coco_names.txt", "r") as f:
    class_names = f.read().strip().split("\n")

# create a list of random colors to represent each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3))

# loop over the frames
while True:
    # starter time to computer the fps
    start = datetime.datetime.now()
    success, frame = video_cap.read()
    h = frame.shape[0]
    w = frame.shape[1]

    # create a blob from the frame
    blob = cv2.dnn.blobFromImage(
        frame, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])
    # pass the blog through our network and get the output predictions
    net.setInput(blob)
    output = net.forward() # shape: (1, 1, 100, 7)

    # loop over the number of detected objects
    for detection in output[0, 0, :, :]: # output[0, 0, :, :] has a shape of: (100, 7)
        # the confidence of the model regarding the detected object
        probability = detection[2]
        # if the confidence of the model is lower than 50%,
        # we do nothing (continue looping)
        if probability < 0.5:
            continue

        # extract the ID of the detected object to get
        # its name and the color associated with it
        class_id = int(detection[1])
        label = class_names[class_id - 1].upper()
        color = colors[class_id]
        B, G, R = int(color[0]), int(color[1]), int(color[2])
        # perform element-wise multiplication to get
        # the (x, y) coordinates of the bounding box
        box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
        box = tuple(box)
        # draw the bounding box of the object
        cv2.rectangle(frame, box[:2], box[2:], (B, G, R), thickness=2)

        # draw the name of the predicted object along with the probability
        text = f"{label} {probability * 100:.2f}%"
        cv2.putText(frame, text, (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    cv2.imshow("Output", frame)
    # write the frame to disk
    writer.write(frame)
    if cv2.waitKey(10) == ord("q"):
        break

# release the video capture, video writer, and close all windows
video_cap.release()
writer.release()
cv2.destroyAllWindows()