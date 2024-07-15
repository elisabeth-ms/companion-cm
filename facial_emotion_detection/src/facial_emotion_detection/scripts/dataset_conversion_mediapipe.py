import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import argparse
import os

FACIAL_LANDMARKS_IDXS = ([
    ("mouth", [0,11,12,13,14,15,16,17,37,39,40,72,38,82,87,86,85,84,73,41,81,178,179,180,181,74,42,80,88,89,90,91,185,184,183,191,95,96,77,146,78,62,76,61,267,302,268,312,317,316,315,314,269,303,271,311,402,403,404,405,270,304,272,310,318,319,320,321,409,408,407,415,324,325,307,375,308,291,306,292,287,432,436,426,423,358,410,322,391,393,164,167,165,92,186,57,43,106,182,83,18,313,406,335,273,212,216,206,203,129,202,204,194,201,200,421,418,424,422]),
    ("right_eye", [133,173,157,158,159,160,161,246,33,7,163,144,145,153,154,155,243,190,56,28,27,29,30,247,130,25,110,24,23,22,26,112,244,221,222,223,224,225,113,226,31,228,229,230,231,232,233,245,193,8,55,65,52,53,46,124,35,9,107,66,105,63,70,156,143,111,117,118,119,120,121,128]),
    ("left_eye", [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,463,341,256,252,253,254,339,255,359,467,260,259,257,258,286,414,464,453,451,451,450,449,448,261,446,342,445,444,443,442,441,413.465,265,353,276,283,282,295,285,8,417,8,285,295,282,283,276,353,9,336,296,334,293,300,383,372,340,346,347,348,349,350,357]),
    ("nose", [2,97,326,328,290,327,460,305,289,392,439,294,331,279,360,344,438,309,250,462,370,94,141,242,20,79,218,115,131,1,4,5,195,197,6,19,125,241,238,354,461,458,459,239,237,44,274,457,220,45,275,440,134,51,281,363,236,3,248,456,174,196,419,399,188,122,351,412,64,240,129,102,49,48,219,166,59,75,235,60,99,203,206,216,358,423,426,436])
])

class FacialEmotionDetector():
    def __init__(self):
        # self.model = YOLO('/home/oem/companion_ws/src/facial_emotion_detection/data/yolov8n-face.pt')
        NUM_FACE = 4

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=1)

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

        # Resize the image
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
        resultant_img = self.landmark_detection(img)

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        # self.draw_fps(img, fps)
        # cv2.imshow('img', resultant_img)

        # while True:
        #     k = cv2.waitKey(30) & 0xff #break when pressing the key "esc"
        #     if k==27:
        #         break

        return resultant_img
    
    def landmark_detection(self, image):
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
                    cv2.circle(image, cord, 1, (0, 255, 0), -1)
                if i in FACIAL_LANDMARKS_IDXS[0][1]:
                    if cord == None:
                        self.mouth.append(0)    
                        self.mouth.append(0)    
                    else:
                        self.mouth.append(cord[0])
                        self.mouth.append(cord[1])
                    cv2.circle(image, cord, 1, (0, 0, 255), -1)
                if i in FACIAL_LANDMARKS_IDXS[3][1]:
                    if cord == None:
                        self.nose.append(0)    
                        self.nose.append(0)    
                    else:
                        self.nose.append(cord[0])   
                        self.nose.append(cord[1])   
                    cv2.circle(image, cord, 1, (255, 0, 0), -1)
                if i in FACIAL_LANDMARKS_IDXS[0][1] or i in FACIAL_LANDMARKS_IDXS[1][1] or i in FACIAL_LANDMARKS_IDXS[2][1] or i in FACIAL_LANDMARKS_IDXS[3][1]:
                    if cord == None:
                        self.total_landmarks.append(0)    
                        self.total_landmarks.append(0)    
                    else:
                        self.total_landmarks.append(cord[0] - cord_centre_nose[0]) 
                        self.total_landmarks.append(cord[1] - cord_centre_nose[1]) 
                    # cv2.circle(image, cord, 1, (255, 255, 0), -1)

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
                    cropped_face = image[y1:y2, x1:x2]
                    # Resize the image
                    cropped_face = cv2.resize(cropped_face, (200, 200))
                    # Detect face landmarks
                    image = self.landmark_detection(cropped_face)
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
