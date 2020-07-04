# Yolo object detection using OpenCV

ref: https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/

## Objective

1. Write program for yolo object detection using OpenCV
2. Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn)
3. Run this image through the program
4. Upload the annotated image by YOLO

## Code

### Pre-requisites

1. Download `yolov3.cfg` from [here](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)
2. Download `coco.names` from [here](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)
3. Download `yolov3.weights` from [here](https://pjreddie.com/media/files/yolov3.weights)

## Results

### Input & Annotated Image

<a href="url"><img src="https://github.com/anuragal/yolov3_opencv/blob/master/IMG.jpg" height="49%" width="49%" ></a>
<a href="url"><img src="https://github.com/anuragal/yolov3_opencv/blob/master/yolo_img_output.jpg" height="49%" width="49%" ></a>

Image Downloaded from [GettyImages](https://www.gettyimages.in/detail/photo/multi-ethnic-business-people-working-together-in-royalty-free-image/468157896)

### Class `YoloV3`

```python
import cv2
import numpy as np

class YoloV3(object):

    def __init__(self):
        self.classes = []
        self.output_layers = None
        self.net = None
        pass

    def load_classes(self, filepath):
        with open(filepath, "r", encoding="utf8") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def load_yolo(self, weights_file, cfg_file):
        self.net = cv2.dnn.readNet(weights_file, cfg_file)
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def process_image(self, input_image):
        img = cv2.imread(input_image)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

        return img

    def show_image(self, img):
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, img, file_path):
        cv2.imwrite(file_path, img)
```

### Run the code `main.py`

```python
from yolov3 import YoloV3

yolo_v3 = YoloV3()
yolo_v3.load_classes("coco.names")
yolo_v3.load_yolo("yolov3.weights", "yolov3.cfg")
img = yolo_v3.process_image("IMG.jpg")
# cv2.imwrite("yolo_img_output.jpg", img)

yolo_v3.show_image(img)
```
