import cv2
import math
import random
from mediapipe.python.solutions import face_mesh

class LivenessService:

    def __init__(self):
        self.mp_face_mesh = face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.challenges = ["BLINK_TWICE", "TURN_LEFT", "TURN_RIGHT"]

        self.BLINK_THRESHOLD = 0.22
        # self.REQUIRED_BLINKS = 1

    def distance(self, pt1, pt2):
        return math.sqrt(
            (pt1.x - pt2.x) ** 2 +
            (pt1.y - pt2.y) ** 2
        )

    def eye_aspect_ratio(self, landmarks, eye_points):
        p1 = landmarks[eye_points[0]]
        p2 = landmarks[eye_points[1]]
        p3 = landmarks[eye_points[2]]
        p4 = landmarks[eye_points[3]]
        p5 = landmarks[eye_points[4]]
        p6 = landmarks[eye_points[5]]

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)
        horizontal = self.distance(p1, p4)

        if horizontal == 0:
            return 0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def verify_liveness(self):

        challenge = self.generate_challenge()
        print(f"Challenge: {challenge}")

        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Could not access webcam.")
            return False

        blink_count = 0
        eye_closed = False
        left_counter = 0
        right_counter = 0
        center_counter = 0
        REQUIRED_DIRECTION_FRAMES = 10

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.putText(img=frame, text=f"Challenge: {challenge}", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                direction = self.detect_head_direction(landmarks)
                # print(direction)

                left_ear = self.eye_aspect_ratio(landmarks, LEFT_EYE)

                right_ear = self.eye_aspect_ratio(landmarks, RIGHT_EYE)

                ear = (left_ear + right_ear) / 2

                cv2.putText(img=frame, text=f"EAR: {ear:.2f}", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

                if ear < self.BLINK_THRESHOLD:
                    eye_closed = True
                else:
                    if eye_closed:
                        blink_count += 1
                        eye_closed = False

                cv2.putText(img=frame, text=f"Blinks: {blink_count}", org=(20, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

                # ========================
                # BLINK CHALLENGE
                # =======================

                if challenge == "BLINK_TWICE":

                    # Wrong head movement
                    if direction in ["LEFT", "RIGHT"]:
                        print("Wrong Challenge Response")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False
                    
                    # elif direction == "CENTER":
                    #     center_counter += 1
                    #     print(f"COUNTER: {center_counter}")
                    #     if center_counter >= 40:
                    #         print("Wrong Challenge Response")
                    #         cap.release()
                    #         cv2.destroyAllWindows()
                    #         return False

                    if blink_count >= 2:
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                    
                # =======================
                # TURN LEFT CHALLENGE
                # =========================
                
                elif challenge == "TURN_LEFT":

                    if direction == "RIGHT":
                        print("Wrong Challenge Response")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False
                    
                    elif direction == "CENTER":
                        center_counter += 1
                        print(f"COUNTER: {center_counter}")
                        if center_counter >= 400:
                            print("Wrong Challenge Response")
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
                    
                    elif direction == "LEFT":
                        left_counter += 1
                        print(f"LEFT COUNTER: {left_counter}")

                    else:
                        left_counter = 0

                    if left_counter >= REQUIRED_DIRECTION_FRAMES:
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                    
                # ==========================
                # TURN RIGHT CHALLENGE
                # ==========================

                elif challenge == "TURN_RIGHT":

                    if direction == "LEFT":
                        print("Wrong Challenge Response")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False
                    
                    elif direction == "CENTER":
                        center_counter += 1
                        print(f"COUNTER: {center_counter}")
                        if center_counter >= 400:
                            print("Wrong Challenge Response")
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
                    
                    elif direction == "RIGHT":
                        right_counter += 1
                        print(f"RIGHT COUNTER: {right_counter}")

                    else:
                        right_counter = 0

                    if right_counter >= REQUIRED_DIRECTION_FRAMES:
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                    
                elif challenge == "TURN_RIGHT":

                    if direction == "RIGHT":
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                
            cv2.imshow("Liveness Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        return False

    def generate_challenge(self):
        return random.choice(self.challenges)

    def detect_head_direction(self, landmarks):

        nose_x = landmarks[1].x
        left_eye_x = landmarks[33].x
        right_eye_x = landmarks[263].x

        face_center = (left_eye_x + right_eye_x) / 2

        diff = nose_x - face_center

        if diff < -0.03:
            return "LEFT"
        
        if diff > 0.03:
            return "RIGHT"
        
        return "CENTER"