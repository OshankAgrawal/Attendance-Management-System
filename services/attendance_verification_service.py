import os
import cv2
import uuid

from services.verification_service import VerificationService
from services.attendance_service import AttendanceService
from services.liveness_service import LivenessService
from services.security_event_service import SecurityEventService

from database.database import SessionLocal
from database.models import Employee


class AttendanceVerificationService:

    def __init__(self):

        self.verifier = VerificationService()
        self.attendance = AttendanceService()
        self.liveness_service = LivenessService()
        self.security_logger = SecurityEventService()

        # Create temp folder if not exists
        os.makedirs("data/temp", exist_ok=True)

    def process_attendance(self, image_path):

        verification_result = (self.verifier.verify_image(image_path))

        if not verification_result.get("verified"):

            return {
                "success": False,
                "message": "Identity Verification Failed",
                "details": verification_result
            }

        db = SessionLocal()

        try:

            employee = (
                db.query(Employee)
                .filter(
                    Employee.employee_id == verification_result["employee_id"]
                )
                .first()
            )

            if employee is None:

                return {
                    "success": False,
                    "message": "Employee not found"
                }

            attendance_result = self.attendance.mark_attendance(employee.id)
            
            liveness_result = self.liveness_service.verify_liveness()

            if not liveness_result:
                self.security_logger.log_event(event_type="LIVENESS_FAILED")
                return{
                    "success": False,
                    "message": "Liveness Check Failed"
                }

            return {

                "success": True,

                "employee_id": employee.employee_id,

                "employee_name": employee.full_name,

                "best_similarity": verification_result["best_similarity"],

                "weighted_average": verification_result["weighted_average"],

                "top3_average": verification_result["top3_average"],

                "attendance": attendance_result
            }

        finally:

            db.close()

    def process_attendance_from_webcam(self):

        print("\nStarting Liveness Check....")

        liveness_result = self.liveness_service.verify_liveness()

        if not liveness_result:
            self.security_logger.log_event(
                event_type="LIVENESS_FAILED",
                description="Challenge Response Failed"
            )
            return {
                "success": False,
                "message": "Liveness Verification Failed"
            }
        
        print("Liveness check Passed")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            return {
                "success": False,
                "message": "Unable to access webcam"
            }

        print("\nPress 'C' to Capture")
        print("Press 'Q' to Quit\n")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("Attendance Verification", frame)

            key = cv2.waitKey(1) & 0xFF

            # Capture Image
            if key == ord("c"):

                image_name = (f"{uuid.uuid4().hex}.jpg")

                image_path = os.path.join("data", "temp", image_name)

                cv2.imwrite(image_path, frame)

                cap.release()
                cv2.destroyAllWindows()

                result = self.process_attendance(image_path)

                return result

            # Quit
            elif key == ord("q"):

                cap.release()
                cv2.destroyAllWindows()

                return {
                    "success": False,
                    "message": "Operation Cancelled"
                }

        cap.release()
        cv2.destroyAllWindows()

        return {
            "success": False,
            "message": "Camera Error"
        }