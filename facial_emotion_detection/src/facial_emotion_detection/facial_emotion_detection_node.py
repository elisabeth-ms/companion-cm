#!/usr/bin/env python3
import rospy
from facial_emotion_detection.facial_emotion_detection_ros_opencv import FacialEmotionDetector

if __name__=='__main__':
    rospy.init_node('facial_emotion_detector')

    facial_emotion_detector = FacialEmotionDetector()

    facial_emotion_detector.start()