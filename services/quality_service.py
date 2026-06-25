import cv2
import numpy as np

class QualityService:

    def __init__(self):
        
        self.MIN_BLUR_SCORE = 100

        self.MIN_DETECTION_SCORE = 0.75

        self.MAX_YAW_ANGLE = 45

    def calculate_blur_score(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        return float(blur_score)
    
    def calculate_pose_score(self, yaw_angle):
        yaw_angle = abs(yaw_angle)

        if yaw_angle > self.MAX_YAW_ANGLE:
            return 0.0
        
        return (self.MAX_YAW_ANGLE - yaw_angle) / self.MAX_YAW_ANGLE
    
    def normalize_blur_score(self, blur_score):
        score = min(blur_score / 300, 1.0)

        return float(score)
    
    def calculate_quality_score(self, blur_score, detection_confidence, pose_score):
        blur_score = self.normalize_blur_score(blur_score)

        quality_score = 0.4 * blur_score + 0.4 * detection_confidence + 0.2 * pose_score

        return round(quality_score, 4)
    
    def is_valid_frame(self, blur_score, detection_confidence, quality_score):
        
        if blur_score < self.MIN_BLUR_SCORE:
            return False
        
        if detection_confidence < self.MIN_DETECTION_SCORE:
            return False
        
        if quality_score <0.80:
            return False
        
        return True