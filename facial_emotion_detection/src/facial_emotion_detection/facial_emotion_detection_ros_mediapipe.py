#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
from ultralytics import YOLO
import time
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from keras.models import load_model
import tensorflow as tf

FACIAL_LANDMARKS_IDXS = ([
    ("mouth", [0,11,12,13,14,15,16,17,37,39,40,72,38,82,87,86,85,84,73,41,81,178,179,180,181,74,42,80,88,89,90,91,185,184,183,191,95,96,77,146,78,62,76,61,267,302,268,312,317,316,315,314,269,303,271,311,402,403,404,405,270,304,272,310,318,319,320,321,409,408,407,415,324,325,307,375,308,291,306,292,287,432,436,426,423,358,410,322,391,393,164,167,165,92,186,57,43,106,182,83,18,313,406,335,273,212,216,206,203,129,202,204,194,201,200,421,418,424,422]),
    # ("right_eyebrow", (17, 21)),
    # ("left_eyebrow", (22, 26)),
    ("right_eye", [133,173,157,158,159,160,161,246,33,7,163,144,145,153,154,155,243,190,56,28,27,29,30,247,130,25,110,24,23,22,26,112,244,221,222,223,224,225,113,226,31,228,229,230,231,232,233,245,193,8,55,65,52,53,46,124,35,9,107,66,105,63,70,156,143,111,117,118,119,120,121,128]),
    ("left_eye", [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,463,341,256,252,253,254,339,255,359,467,260,259,257,258,286,414,464,453,451,451,450,449,448,261,446,342,445,444,443,442,441,413.465,265,353,276,283,282,295,285,8,417,8,285,295,282,283,276,353,9,336,296,334,293,300,383,372,340,346,347,348,349,350,357]),
    ("nose", [2,97,326,328,290,327,460,305,289,392,439,294,331,279,360,344,438,309,250,462,370,94,141,242,20,79,218,115,131,1,4,5,195,197,6,19,125,241,238,354,461,458,459,239,237,44,274,457,220,45,275,440,134,51,281,363,236,3,248,456,174,196,419,399,188,122,351,412,64,240,129,102,49,48,219,166,59,75,235,60,99,203,206,216,358,423,426,436]),
    # ("jaw", [0, 16])
])

def getLabel(id):
    return ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'][id]

def imgmsg_to_cv2(img_msg):
    # if img_msg.encoding != "bgr8":
    #     rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tobytes()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

class FacialEmotionDetector():
    def __init__(self):
        self.read_params()

    def read_params(self):
        self.image_topic = rospy.get_param("~/facial_emotion_detector/input_image_topic")
        topic_list = self.image_topic.split("/")
        self.output_topic = "/".join(topic_list[:-1]) + "/facial_emotion"

    def ros_setup(self):
        self.model = YOLO('/home/oem/companion_ws/src/facial_emotion_detection/data/yolov8n-face.pt')
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_cb)
        self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.facemark = cv2.face.createFacemarkLBF()
        cv2.face_Facemark.loadModel(self.facemark, '/home/oem/companion_ws/src/facial_emotion_detection/data/lbfmodel.yaml')
        pTime = 0
        NUM_FACE = 4

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=1)

        self.emotion_detection_model = load_model('/home/oem/companion_ws/src/facial_emotion_detection/data/model_total_landmarks.h5')

    def start(self):
        self.ros_setup()
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def image_cb(self, data):
        cv_image = imgmsg_to_cv2(data)
        computed_image, results = self.face_detection(cv_image)
        # self.landmark_detection(computed_image, results)
        self.image_pub.publish(cv2_to_imgmsg(computed_image))
        return
        
    def face_detection(self, img):
        prevTime = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Run inference
        results = self.model(source=img, conf=0.25, device='cuda')
        # Draw the bboxes
        resultant_img = self.draw_bboxes(img, results)
        currTime = time.time()
        fps = 1/(currTime-prevTime)
        print(fps)
        self.draw_fps(img, fps)

        return resultant_img, results
    
    def landmark_detection(self, image, x1, y1, x2, y2):
        self.mouth = []
        self.eyes = []
        self.nose = []
        self.total_landmarks = []
        image_rows, image_cols, _ = image.shape
        # Convert the input image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Estimate facial landmarks
        results = self.faceMesh.process(image)
        if results.multi_face_landmarks is not None:
            ls_single_face=results.multi_face_landmarks[0].landmark
            for i, idx in enumerate(ls_single_face):
                cord = _normalized_to_pixel_coordinates(idx.x,idx.y,image_cols,image_rows) 
                if i == 1:
                    cord_centre_nose = cord
            for i, idx in enumerate(ls_single_face):
                # print(idx)   
                cord = _normalized_to_pixel_coordinates(idx.x,idx.y,image_cols,image_rows)                    
                if i in FACIAL_LANDMARKS_IDXS[1][1] or i in FACIAL_LANDMARKS_IDXS[2][1]:             
                    if cord == None:
                        self.eyes.append(0)    
                        self.eyes.append(0)   
                    else:
                        self.eyes.append(cord[0])
                        self.eyes.append(cord[1])
                    # cv2.circle(image, cord, 1, (0, 255, 0), 3)
                if i in FACIAL_LANDMARKS_IDXS[0][1]:
                    if cord == None:
                        self.mouth.append(0)    
                        self.mouth.append(0)    
                    else:
                        self.mouth.append(cord[0])
                        self.mouth.append(cord[1])
                    # cv2.circle(image, cord, 1, (0, 0, 255), 3)
                if i in FACIAL_LANDMARKS_IDXS[3][1]:
                    if cord == None:
                        self.nose.append(0)    
                        self.nose.append(0)    
                    else:
                        self.nose.append(cord[0])   
                        self.nose.append(cord[1])   
                    # cv2.circle(image, cord, 1, (255, 0, 0), 3)
                if i in FACIAL_LANDMARKS_IDXS[0][1] or i in FACIAL_LANDMARKS_IDXS[1][1] or i in FACIAL_LANDMARKS_IDXS[2][1] or i in FACIAL_LANDMARKS_IDXS[3][1]:
                    if cord == None:
                        self.total_landmarks.append(0)    
                        self.total_landmarks.append(0)    
                    else:
                        self.total_landmarks.append(cord[0] - cord_centre_nose[0]) 
                        self.total_landmarks.append(cord[1] - cord_centre_nose[1]) 
                    cv2.circle(image, cord, 1, (255, 255, 0), 3)
                # print(self.total_landmarks)
        self.total_landmarks = np.array(self.total_landmarks)
        self.total_landmarks = self.total_landmarks.reshape(1,678)
        ynew = self.emotion_detection_model.predict(self.total_landmarks)
        im_class = tf.argmax(ynew[0], axis=-1)
        emotion = getLabel(im_class)
        return image, emotion
            
    def draw_bboxes(self, image, detections):
        for detection in detections:
            if len(detection) == 0:
                return image
            else:
                bbox_xyxy = detection.boxes.xyxy
                for i, box in enumerate(bbox_xyxy):               
                    x1, y1, x2, y2 = [int(i) for i in box]
                    # print((x1, y1, x2, y2))
                    c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(image, c1, c2, (0,0,255), thickness=2, lineType=cv2.LINE_AA)
                    # Crop the region of interest from the image
                    cropped_face = image[y1:y2, x1:x2]
                    # Resize the image
                    cropped_face = cv2.resize(cropped_face, (640, 640), interpolation=cv2.INTER_AREA)
                    # Detect face landmarks
                    image, emotion = self.landmark_detection(cropped_face, x1, y1, x2, y2)
                # Write the Confidence of the detection above the bbox
                line_thickness = 3
                tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
                tf = max(tl - 1, 1)  # font thickness
                # t_size = cv2.getTextSize(f'Conf:{str(round(float(detection.boxes.conf[0]),2))}', 0, fontScale=tl / 3, thickness=tf)[0]
                t_size = cv2.getTextSize(f'Emotion:{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (0,0,255), 1, 8, 2)
                # cv2.putText(image, f'Conf:{str(round(float(detection.boxes.conf[0]),2))}', (c1[0], c1[1] - 2), 0, tl / 3,
                #             [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA) 
                cv2.putText(image, f'Emotion:{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
                            [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)        
                return image
    
    def draw_border(self, img, pt1, pt2, color, thickness, r, d):
        x1, y1 = pt1
        x2, y2 = pt2
        # Top leftfrom collections import deque (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

        cv2.circle(img, (x1 + r, y1+r), 2, color, 12)
        cv2.circle(img, (x2 - r, y1+r), 2, color, 12)
        cv2.circle(img, (x1 + r, y2-r), 2, color, 12)
        cv2.circle(img, (x2 - r, y2-r), 2, color, 12)
        return

    def draw_fps(self, img, fps):
        cv2.line(img, (20, 25), (150, 25), [85, 45, 255], 30)
        cv2.putText(img, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                    225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        return img
        

        
        
