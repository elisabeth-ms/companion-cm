#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
from ultralytics import YOLO
import time
import pybboxes as pbx
import dlib
from imutils import face_utils
from keras.models import load_model
import tensorflow as tf
import rospkg
import os

FACIAL_LANDMARKS_IDXS = ([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 21)),
    ("left_eyebrow", (22, 26)),
    ("right_eye", (36, 41)),
    ("left_eye", (42, 47)),
    ("eyes", (36, 47)),
    ("eyebrows", (17, 26)),
    ("nose", (27, 35)),
    ("jaw", (0, 16))
])

def getLabel(id):
    return ['neutral', 'anger', 'sadness', 'happiness', 'fear', 'contempt', 'surprise', 'disgust'][id]

# ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

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
        # self.mouth_min, self.mouth_max = FACIAL_LANDMARKS_IDXS[0][1]
        # self.eyes_min, self.eyes_max = FACIAL_LANDMARKS_IDXS[5][1]
        # self.eyebrows_min, self.eyebrows_max = FACIAL_LANDMARKS_IDXS[6][1]
        # self.nose_min, self.nose_max = FACIAL_LANDMARKS_IDXS[7][1]

        self.mouth_min, self.mouth_max = FACIAL_LANDMARKS_IDXS[0][1]
        self.eyes_min, self.eyes_max = FACIAL_LANDMARKS_IDXS[5][1]
        self.eyebrows_min, self.eyebrows_max = FACIAL_LANDMARKS_IDXS[6][1]
        self.nose_min, self.nose_max = FACIAL_LANDMARKS_IDXS[7][1]
        self.left_eyebrow_min, self.left_eyebrow_max = FACIAL_LANDMARKS_IDXS[2][1]
        self.right_eyebrow_min, self.right_eyebrow_max = FACIAL_LANDMARKS_IDXS[1][1]
        self.right_eye_min, self.right_eye_max = FACIAL_LANDMARKS_IDXS[3][1]
        self.left_eye_min, self.left_eye_max = FACIAL_LANDMARKS_IDXS[4][1]
        self.jaw_min, self.jaw_max = FACIAL_LANDMARKS_IDXS[8][1]

        self.emotion_past = 'contempt'

    def read_params(self):
        self.image_topic = rospy.get_param("~/facial_emotion_detector/input_image_topic")
        topic_list = self.image_topic.split("/")
        self.output_topic = "/".join(topic_list[:-1]) + "/facial_emotion"

    def ros_setup(self):

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('facial_emotion_detection')

        yolo_model_path = os.path.join(package_path, 'data', 'yolov8n-face.pt')
        self.model = YOLO('/home/oem/companion_ws/src/facial_emotion_detection/data/yolov8n-face.pt')
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_cb)
        self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.facemark = cv2.face.createFacemarkLBF()
        cv2.face_Facemark.loadModel(self.facemark, '/home/oem/companion_ws/src/facial_emotion_detection/data/lbfmodel.yaml')

        # dlib
        dlib_path = os.path.join(package_path, 'data', 'shape_predictor_68_face_landmarks_GTX.dat')
        # Model_PATH = '/home/oem/companion_ws/src/facial_emotion_detection/data/shape_predictor_68_face_landmarks_GTX.dat'
        self.faceLandmarkDetector = dlib.shape_predictor(dlib_path)
        
        emotion_detection_model_path = os.path.join(package_path, 'data', 'model_total_landmarks1.h5')
        self.emotion_detection_model = load_model(emotion_detection_model_path)
        # self.emotion_detection_model = load_model('/home/oem/companion_ws/src/facial_emotion_detection/data/model_total_landmarks1.h5')

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
        results = self.model(source=img, conf=0.45)
        # Draw the bboxes
        resultant_img = self.draw_bboxes(img, results)
        currTime = time.time()
        fps = 1/(currTime-prevTime)
        self.draw_fps(img, fps)

        return resultant_img, results
    
    def landmark_detection(self, image, x1, y1, x2, y2):
        self.mouth = []
        self.eyes = []
        self.nose = []
        self.total_landmarks = []
        # Convert the input image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bbox = cv2.UMat(np.array([(x1, y1, x2-x1, y2-y1)]))  # Convert the bounding box to a cv::UMat
        # Estimate facial landmarks
        retVal, landmarks = self.facemark.fit(gray, bbox)
        for landmark in landmarks:            
            landmark_np = landmark.get()
            for idx, landmark in enumerate(landmark_np[0]):
                if idx >= self.mouth_min and idx <= self.mouth_max:    
                    # Save the two points for later calculating the centre point of the mouth 
                    if idx == 48:
                        L1_mouth_x = landmark[0]
                        L1_mouth_y = landmark[1]
                    if idx == 54:
                        L2_mouth_x = landmark[0]
                        L2_mouth_y = landmark[1]
                if idx >= self.nose_min and idx <= self.nose_max:    
                    # Save the center point of the nose 
                    if idx == 30:
                        C_nose_x = landmark[0]
                        C_nose_y = landmark[1]
                if idx >= self.eyes_min and idx <= self.eyes_max or idx >= self.eyebrows_min and idx <= self.eyebrows_max:    
                    # Save the four points for later calculating the centre point of the left and right eyes 
                    if idx == 36:
                        L1_r_eye_x = landmark[0]
                        L1_r_eye_y = landmark[1]
                    if idx == 39:
                        L2_r_eye_x = landmark[0]
                        L2_r_eye_y = landmark[1]
                    if idx == 42:
                        L1_l_eye_x = landmark[0]
                        L1_l_eye_y = landmark[1]
                    if idx == 45:
                        L2_l_eye_x = landmark[0]
                        L2_l_eye_y = landmark[1]

        # Calculate mouth and eyes center points
        C_re_x = (L1_r_eye_x + L2_r_eye_x) / 2
        C_re_y = (L1_r_eye_y + L2_r_eye_y) / 2

        C_le_x = (L1_l_eye_x + L2_l_eye_x) / 2
        C_le_y = (L1_l_eye_y + L2_l_eye_y) / 2

        C_mouth_x = (L1_mouth_x + L2_mouth_x) / 2
        C_mouth_y = (L1_mouth_y + L2_mouth_y) / 2

        for landmark in landmarks:            
            landmark_np = landmark.get()
            for idx, landmark in enumerate(landmark_np[0]):
                if idx >= self.mouth_min and idx <= self.mouth_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (255, 0, 0), -1)
                    self.mouth.append(round(landmark[0] - C_mouth_x, 6)) 
                    self.mouth.append(round(landmark[1] - C_mouth_y, 6))

                    self.total_landmarks.append(round(landmark[0] - C_mouth_x, 6)) 
                    self.total_landmarks.append(round(landmark[1] - C_mouth_y, 6))

                if idx >= self.nose_min and idx <= self.nose_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 255, 0), -1)
                    self.nose.append(round(landmark[0] - C_nose_x, 6)) 
                    self.nose.append(round(landmark[1] - C_nose_y, 6))

                    self.total_landmarks.append(round(landmark[0] - C_nose_x, 6)) 
                    self.total_landmarks.append(round(landmark[1] - C_nose_y, 6))

                if idx >= self.right_eye_min and idx <= self.right_eye_max or idx >= self.right_eyebrow_min and idx <= self.right_eyebrow_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                    self.eyes.append(round(landmark[0] - C_re_x, 6)) 
                    self.eyes.append(round(landmark[1] - C_re_y, 6))

                    self.total_landmarks.append(round(landmark[0] - C_re_x, 6)) 
                    self.total_landmarks.append(round(landmark[1] - C_re_y, 6))
                
                if idx >= self.left_eye_min and idx <= self.left_eye_max or idx >= self.left_eyebrow_min and idx <= self.left_eyebrow_max:   
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                    self.eyes.append(round(landmark[0] - C_le_x, 6))
                    self.eyes.append(round(landmark[1] - C_le_y, 6))

                    self.total_landmarks.append(round(landmark[0] - C_le_x, 6)) 
                    self.total_landmarks.append(round(landmark[1] - C_le_y, 6))

                if idx >= self.jaw_min and idx <= self.jaw_max:
                    self.total_landmarks.append(round(landmark[0] - C_nose_x, 6))
                    self.total_landmarks.append(round(landmark[1] - C_nose_y, 6))

        self.total_landmarks = np.array(self.total_landmarks)
        self.eyes = np.array(self.eyes)
        self.nose = np.array(self.nose)
        self.mouth = np.array(self.mouth)
        self.total_landmarks = self.total_landmarks.reshape(1,136)
        self.eyes = self.eyes.reshape(1,44)
        self.nose = self.nose.reshape(1,18)
        self.mouth = self.mouth.reshape(1,40)
        ynew = self.emotion_detection_model.predict(self.total_landmarks)
        # ynew = self.emotion_detection_model.predict([self.eyes, self.nose, self.mouth])
        print(ynew[0])
        im_class = tf.argmax(ynew[0], axis=-1)
        
        emotion = getLabel(im_class)

        print(emotion)

        return image, emotion
            
    def draw_bboxes(self, image, detections):
        for detection in detections:
            if len(detection) == 0:
                return image
            else:
                bbox_xyxy = detection.boxes.xyxy
                for i, box in enumerate(bbox_xyxy):               
                    x1, y1, x2, y2 = [int(i) for i in box]
                    print((x1, y1, x2, y2))
                    c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
                    # cv2.rectangle(image, c1, c2, (0,0,255), thickness=2, lineType=cv2.LINE_AA)
                    # Crop the region of interest from the image
                    cropped_face = image[y1:y2, x1:x2]
                    # cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
                    # Resize the image
                    # print(cropped_face.shape)
                    cropped_face = cv2.resize(cropped_face, (200, 200), interpolation=cv2.INTER_AREA)
                    cropped_face, emotion = self.landmark_detection(cropped_face, 0, 0, cropped_face.shape[0] - 1, cropped_face.shape[1] - 1)
                    # image, emotion = self.landmark_detection(image, x1, y1, x2, y2)
                # Write the Confidence of the detection above the bbox
                line_thickness = 3
                tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
                tf = max(tl - 1, 1)  # font thickness
                if emotion == 'happiness':
                    cv2.rectangle(image, c1, c2, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
                    t_size = cv2.getTextSize(f'{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                    self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (0,255,0), 1, 8, 2)
                    cv2.putText(image, f'{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
                                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)    
                elif emotion == 'anger':
                    cv2.rectangle(image, c1, c2, (0,0,255), thickness=2, lineType=cv2.LINE_AA)
                    t_size = cv2.getTextSize(f'{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                    self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (0,0,255), 1, 8, 2)
                    cv2.putText(image, f'{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
                                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)     
                elif emotion == 'surprise':
                    cv2.rectangle(image, c1, c2, (128,0,128), thickness=2, lineType=cv2.LINE_AA)
                    t_size = cv2.getTextSize(f'{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                    self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (128,0,128), 1, 8, 2)
                    cv2.putText(image, f'{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
                                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)     
                elif emotion == 'contempt':
                    cv2.rectangle(image, c1, c2, (128,128,128), thickness=2, lineType=cv2.LINE_AA)
                    t_size = cv2.getTextSize(f'{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                    self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (128,128,128), 1, 8, 2)
                    cv2.putText(image, f'{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
                                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)    
                elif emotion == 'fear':
                    cv2.rectangle(image, c1, c2, (255,0,0), thickness=2, lineType=cv2.LINE_AA)
                    t_size = cv2.getTextSize(f'{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                    self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (255,0,0), 1, 8, 2)
                    cv2.putText(image, f'{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
                                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                elif emotion == 'disgust':
                    cv2.rectangle(image, c1, c2, (0,255,255), thickness=2, lineType=cv2.LINE_AA)
                    t_size = cv2.getTextSize(f'{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                    self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (0,255,255), 1, 8, 2)
                    cv2.putText(image, f'{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
                                [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
                elif emotion == 'sadness':
                    cv2.rectangle(image, c1, c2, (128,128,255), thickness=2, lineType=cv2.LINE_AA)
                    t_size = cv2.getTextSize(f'{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                    self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (128,128,0), 1, 8, 2)
                    cv2.putText(image, f'{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
                                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                else:
                    cv2.rectangle(image, c1, c2, (128,255,128), thickness=2, lineType=cv2.LINE_AA)
                    t_size = cv2.getTextSize(f'{emotion}', 0, fontScale=tl / 3, thickness=tf)[0]
                    self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (128,128,0), 1, 8, 2)
                    cv2.putText(image, f'{emotion}', (c1[0], c1[1] - 2), 0, tl / 3,
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
        

        
        
