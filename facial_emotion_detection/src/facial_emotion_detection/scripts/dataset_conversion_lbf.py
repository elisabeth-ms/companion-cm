import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import argparse
import os
from ultralytics import YOLO

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

class FacialEmotionDetector():
    def __init__(self):
        self.model = YOLO('/home/oem/companion_ws/src/facial_emotion_detection/data/yolov8n-face.pt')
        NUM_FACE = 4

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=1)
        self.facemark = cv2.face.createFacemarkLBF()
        cv2.face_Facemark.loadModel(self.facemark, '/home/oem/companion_ws/src/facial_emotion_detection/data/lbfmodel.yaml')

    def detect(self, folder_path):
        total_img=[]
        total_dirs=[]

        for root, dirs, files in os.walk(folder_path):           
            file_paths = [os.path.join(root, f) for f in files]
            total_img += file_paths
            total_dirs+=(dirs)

        total_dirs = set(total_dirs)

        for img_path in total_img:
            self.mouth = []
            self.eyes = []
            self.nose = []
            self.total_landmarks = []

            print(img_path)
            cv_image = cv2.imread(img_path)
            computed_image = self.face_detection(cv_image)
        
            txt_file_name = f'{self.get_last_word_without_extention(img_path)}.txt'

            if not os.path.exists(f'{os.path.dirname(os.path.dirname(img_path))}/labels/'):
                os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/')
                os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/total/')
                os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/eyes/')
                os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/nose/')
                os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/mouth/')
                for dir in total_dirs:
                    os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/total/{dir}/')
                    os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/eyes/{dir}/')
                    os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/nose/{dir}/')
                    os.mkdir(f'{os.path.dirname(os.path.dirname(img_path))}/labels/mouth/{dir}/')
 
            for dir in total_dirs:
                if os.path.split(os.path.dirname(img_path))[-1] == dir:
                    if os.path.isfile(f'{os.path.dirname(os.path.dirname(img_path))}/labels/total/{dir}/{txt_file_name}'):
                        continue
                    with open(f'{os.path.dirname(os.path.dirname(img_path))}/labels/total/{dir}/{txt_file_name}', 'w+') as f:
                        for item in self.total_landmarks:
                            f.write('%s\n' % item)
                        f.close()
                    if os.path.isfile(f'{os.path.dirname(os.path.dirname(img_path))}/labels/eyes/{dir}/{txt_file_name}'):
                        continue
                    with open(f'{os.path.dirname(os.path.dirname(img_path))}/labels/eyes/{dir}/{txt_file_name}', 'w+') as f:
                        for item in self.eyes:
                            f.write('%s\n' % item)
                        f.close()
                    if os.path.isfile(f'{os.path.dirname(os.path.dirname(img_path))}/labels/nose/{dir}/{txt_file_name}'):
                        continue
                    with open(f'{os.path.dirname(os.path.dirname(img_path))}/labels/nose/{dir}/{txt_file_name}', 'w+') as f:
                        for item in self.nose:
                            f.write('%s\n' % item)
                        f.close()
                    if os.path.isfile(f'{os.path.dirname(os.path.dirname(img_path))}/labels/mouth/{dir}/{txt_file_name}'):
                        continue
                    with open(f'{os.path.dirname(os.path.dirname(img_path))}/labels/mouth/{dir}/{txt_file_name}', 'w+') as f:
                        for item in self.mouth:
                            f.write('%s\n' % item)
                        f.close()
        
    def face_detection(self, img):
        prevTime = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # # Resize the image
        # img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        # resultant_img = self.landmark_detection(img, 0, 0, img.shape[0] - 1, img.shape[1] - 1)

        # Run inference
        results = self.model(source=img, conf=0.45)
        # Draw the bboxes
        resultant_img = self.draw_bboxes(img, results)

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        # self.draw_fps(img, fps)
        # cv2.imshow('img', resultant_img)

        # while True:
        #     k = cv2.waitKey(30) & 0xff #break when pressing the key "esc"
        #     if k==27:
        #         break

        return resultant_img
    
    def landmark_detection(self, image, x1, y1, x2, y2):
         # Convert the input image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bbox = cv2.UMat(np.array([(x1, y1, x2, y2)]))  # Convert the bounding box to a cv::UMat
        # Estimate facial landmarks
        retVal, landmarks = self.facemark.fit(gray, bbox)

        mouth_min, mouth_max = FACIAL_LANDMARKS_IDXS[0][1]
        eyes_min, eyes_max = FACIAL_LANDMARKS_IDXS[5][1]
        eyebrows_min, eyebrows_max = FACIAL_LANDMARKS_IDXS[6][1]
        nose_min, nose_max = FACIAL_LANDMARKS_IDXS[7][1]
        left_eyebrow_min, left_eyebrow_max = FACIAL_LANDMARKS_IDXS[2][1]
        right_eyebrow_min, right_eyebrow_max = FACIAL_LANDMARKS_IDXS[1][1]
        right_eye_min, right_eye_max = FACIAL_LANDMARKS_IDXS[3][1]
        left_eye_min, left_eye_max = FACIAL_LANDMARKS_IDXS[4][1]
        jaw_min, jaw_max = FACIAL_LANDMARKS_IDXS[8][1]

        for landmark in landmarks:            
            landmark_np = landmark.get()
            for idx, landmark in enumerate(landmark_np[0]):
                if idx >= mouth_min and idx <= mouth_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (255, 0, 0), -1)
                    # self.mouth.append(landmark[0]) 
                    # self.mouth.append(landmark[1])
                    # Save the two points for later calculating the centre point of the mouth 
                    if idx == 48:
                        # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (255, 0, 0), -1)
                        L1_mouth_x = landmark[0]
                        L1_mouth_y = landmark[1]
                    if idx == 54:
                        # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (255, 0, 0), -1)
                        L2_mouth_x = landmark[0]
                        L2_mouth_y = landmark[1]
                if idx >= nose_min and idx <= nose_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 255, 0), -1)
                    # self.nose.append(landmark[0]) 
                    # self.nose.append(landmark[1])
                    # Save the center point of the nose 
                    if idx == 30:
                        # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 255, 0), -1)
                        C_nose_x = landmark[0]
                        C_nose_y = landmark[1]
                if idx >= eyes_min and idx <= eyes_max or idx >= eyebrows_min and idx <= eyebrows_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                    # self.eyes.append(landmark[0]) 
                    # self.eyes.append(landmark[1])
                    # Save the four points for later calculating the centre point of the left and right eyes 
                    if idx == 36:
                        # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                        L1_r_eye_x = landmark[0]
                        L1_r_eye_y = landmark[1]
                    if idx == 39:
                        # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                        L2_r_eye_x = landmark[0]
                        L2_r_eye_y = landmark[1]
                    if idx == 42:
                        # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                        L1_l_eye_x = landmark[0]
                        L1_l_eye_y = landmark[1]
                    if idx == 45:
                        # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                        L2_l_eye_x = landmark[0]
                        L2_l_eye_y = landmark[1]

                cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 255, 255), -1)
                # self.total_landmarks.append(landmark[0]) 
                # self.total_landmarks.append(landmark[1])

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
                if idx >= mouth_min and idx <= mouth_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (255, 0, 0), -1)
                    self.mouth.append(round(landmark[0] - C_mouth_x, 6)) 
                    self.mouth.append(round(landmark[1] - C_mouth_y, 6))

                    self.total_landmarks.append(round(landmark[0] - C_mouth_x, 6)) 
                    self.total_landmarks.append(round(landmark[1] - C_mouth_y, 6))

                if idx >= nose_min and idx <= nose_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 255, 0), -1)
                    self.nose.append(round(landmark[0] - C_nose_x, 6)) 
                    self.nose.append(round(landmark[1] - C_nose_y, 6))

                    self.total_landmarks.append(round(landmark[0] - C_nose_x, 6)) 
                    self.total_landmarks.append(round(landmark[1] - C_nose_y, 6))

                if idx >= right_eye_min and idx <= right_eye_max or idx >= right_eyebrow_min and idx <= right_eyebrow_max:    
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                    self.eyes.append(round(landmark[0] - C_re_x, 6)) 
                    self.eyes.append(round(landmark[1] - C_re_y, 6))

                    self.total_landmarks.append(round(landmark[0] - C_re_x, 6)) 
                    self.total_landmarks.append(round(landmark[1] - C_re_y, 6))
                
                if idx >= left_eye_min and idx <= left_eye_max or idx >= left_eyebrow_min and idx <= left_eyebrow_max:   
                    # Draw the facial landmarks on the original image                
                    # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 0, 255), -1)
                    self.eyes.append(round(landmark[0] - C_le_x, 6))
                    self.eyes.append(round(landmark[1] - C_le_y, 6))

                    self.total_landmarks.append(round(landmark[0] - C_le_x, 6)) 
                    self.total_landmarks.append(round(landmark[1] - C_le_y, 6))

                if idx >= jaw_min and idx <= jaw_max:
                    self.total_landmarks.append(round(landmark[0] - C_nose_x, 6))
                    self.total_landmarks.append(round(landmark[1] - C_nose_y, 6))

                # cv2.circle(image, ((int(landmark[0])), (int(landmark[1]))), 3, (0, 255, 255), -1)

        return image
            
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
                    # cv2.rectangle(image, c1, c2, (0,0,255), thickness=2, lineType=cv2.LINE_AA)
                    # Crop the region of interest from the image
                    cropped_face = image[y1-20:y2+20, x1-20:x2+20]
                    # Resize the image
                    # cropped_face = cv2.resize(cropped_face, (200, 200))
                    cropped_face = cv2.resize(cropped_face, (200, 200), interpolation=cv2.INTER_AREA)
                    # Detect face landmarks
                    image = self.landmark_detection(cropped_face, 0, 0, cropped_face.shape[0] - 1, cropped_face.shape[1] - 1)
                    # image = self.landmark_detection(image, x1, y1, x2, y2)
                # Write the Confidence of the detection above the bbox
                line_thickness = 3
                tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(f'Conf:{str(round(float(detection.boxes.conf[0]),2))}', 0, fontScale=tl / 3, thickness=tf)[0]
                # self.draw_border(image, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]+3), (0,0,255), 1, 8, 2)
                # cv2.putText(image, f'Conf:{str(round(float(detection.boxes.conf[0]),2))}', (c1[0], c1[1] - 2), 0, tl / 3,
                #             [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)        
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
    
    def get_string_without_extention(self, s):
        return s[:s.index('.')]

    def get_last_word_in_a_path(self, s):
        return s.split(os.sep)[-1]

    def get_last_word_without_extention(self, s):
        return self.get_string_without_extention(self.get_last_word_in_a_path(s))
        

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    # Add arguments
    parser.add_argument('-f', '--folder_images_path', type=str, help='Folder path of the images.')
    # Parse the command-line arguments
    args = parser.parse_args()
    # Access the values of the arguments
    folder_path = args.folder_images_path

    facial_landmark_detector = FacialEmotionDetector()

    facial_landmark_detector.detect(folder_path)
