
from yolov3 import YoloV3

yolo_v3 = YoloV3()
yolo_v3.load_classes("coco.names")
yolo_v3.load_yolo("yolov3.weights", "yolov3.cfg")
img = yolo_v3.process_image("IMG.jpg")
# cv2.imwrite("yolo_img_output.jpg", img)

yolo_v3.show_image(img)

yolo_v3.save_image(img, "yolo_img_output.jpg")

