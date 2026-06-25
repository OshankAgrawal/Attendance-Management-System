import cv2
import os

class FaceService:

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

    def create_user_directory(self, employee_id):

        path = os.path.join("data", "user_images", employee_id)

        os.makedirs(path, exist_ok=True)

        return path

    def capture_faces(self, employee_id, num_images=20):

        save_path = self.create_user_directory(employee_id)

        cap = cv2.VideoCapture(0)

        count = 0

        print("\nStarting Face Enrollment...")
        print("Press Q to Quit")

        while True:

            success, frame = cap.read()

            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
                count += 1
                image_path = os.path.join(save_path, f"{employee_id}_{count}.jpg")

                cv2.imwrite(image_path,frame)

                print(f"Captured {count}/{num_images}")

                cv2.waitKey(300)

                if count >= num_images:
                    break

            cv2.putText(img=frame, text=f"Captured: {count}/{num_images}", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2, color=(0, 255, 0), fontScale=1)

            cv2.imshow("Face Enrollment",frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if count >= num_images:
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\nEnrollment Complete")

        return save_path