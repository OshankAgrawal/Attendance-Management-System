import os
import cv2

from database.database import SessionLocal
from database.models import Employee, FaceImage, FaceEmbedding, AuditLog

from services.face_service import FaceService
from services.embedding_service import EmbeddingService
from services.quality_service import QualityService


class EnrollmentService:

    def __init__(self):

        self.face_service = FaceService()

        self.embedding_service = EmbeddingService()

        self.quality_service = QualityService()

    # def enroll_employee(self, employee_id, full_name, department, designation=None, email=None):

    #     db = SessionLocal()

    #     try:

    #         # -------------------------
    #         # Create Employee
    #         # -------------------------

    #         employee = Employee(
    #             employee_id=employee_id,
    #             full_name=full_name,
    #             department=department,
    #             designation=designation,
    #             email=email
    #         )

    #         db.add(employee)
    #         db.commit()
    #         db.refresh(employee)

    #         print(f"Employee Created: {employee_id}")

    #         # -------------------------
    #         # Capture Images
    #         # -------------------------

    #         image_folder = self.face_service.capture_faces(employee_id=employee_id, num_images=20)

    #         image_files = sorted(os.listdir(image_folder))

    #         # -------------------------
    #         # Generate Embeddings
    #         # -------------------------

    #         embedding_count = 0

    #         for image_name in image_files:

    #             image_path = os.path.join(image_folder, image_name)

    #             face_image = FaceImage(employee_id=employee.id, image_path=image_path)

    #             db.add(face_image)
    #             db.commit()
    #             db.refresh(face_image)

    #             face_data = self.embedding_service.generate_face_data(image_path)

    #             if face_data is None:
    #                 print(f"Skipping: {image_name}")
    #                 continue

    #             image = cv2.imread(image_path)

    #             blur_score = self.quality_service.calculate_blur_score(image)

    #             detection_confidence = face_data["detection_confidence"]

    #             # Temporary
    #             pose_score = 1.0

    #             quality_score = self.quality_service.calculate_quality_score(blur_score, detection_confidence, pose_score)

    #             is_valid = self.quality_service.is_valid_frame(blur_score, detection_confidence, quality_score)
                
    #             print(f"Blur={blur_score:.2f} ", f"Confidence={detection_confidence:.2f} ", f"Quality_Score={quality_score:.2f}")

    #             if not is_valid:
    #                 print(f"Rejected: {image_name} | ", f"Blur={blur_score:.2f} ", f"Confidennce={detection_confidence:.2f} ", f"Quality={quality_score:.2f}")
    #                 continue

    #             face_embedding = FaceEmbedding(
    #                     employee_id=employee.id,
    #                     image_id=face_image.id,
    #                     embedding=face_data["embedding"],
    #                     pose_type="UNKNOWN",
    #                     quality_score=quality_score,
    #                     blur_score=blur_score,
    #                     detection_confidence=detection_confidence,
    #                     yaw_angle=0.0,
    #                     model_name="arcface"
    #                 )
                
    #             print("Embedding Lenght: ", len(face_data["embedding"]))

    #             db.add(face_embedding)

    #             embedding_count += 1

    #         db.commit()

    #         # -------------------------
    #         # Audit Log
    #         # -------------------------

    #         log = AuditLog(
    #             action="EMPLOYEE_ENROLLED",
    #             details=(
    #                 f"{employee_id} enrolled "
    #                 f"with {embedding_count} "
    #                 f"embeddings"
    #             )
    #         )

    #         db.add(log)

    #         db.commit()

    #         print(f"Enrollment Completed ", f"({embedding_count} embeddings)")

    #         return True

    #     except Exception as e:

    #         db.rollback()

    #         print("Error:", e)

    #         return False

    #     finally:

    #         db.close()

    def start_face_enrollment(self, employee_db_id):

        db = SessionLocal()

        try:
            employee = db.query(Employee).filter(Employee.id == employee_db_id).first()

            if not employee:
                return {
                    "success": False,
                    "message": "Employee not found"
                }
            
            print(f"Starting Enrollment For: {employee.employee_id}")

            # ========================
            # Capture Images
            # ========================

            image_folder = self.face_service.capture_faces(
                employee_id=employee.employee_id,
                num_images=20
            )

            image_files = sorted(os.listdir(image_folder))

            embedding_count = 0
            valid_frame_count = 0

            # ============================
            # Generate Embeddings
            # ============================
            
            for image_name in image_files:

                image_path = os.path.join(image_folder, image_name)

                face_image = FaceImage(employee_id = employee_db_id, image_path=image_path)

                db.add(face_image)
                db.commit()
                db.refresh(face_image)

                face_data = self.embedding_service.generate_face_data(image_path)

                if face_data is None:
                    print(f"Skipping {image_name}", f"(No Face Found)")
                    continue

                image = cv2.imread(image_path)

                blur_score = self.quality_service.calculate_blur_score(image)

                detection_confidence = face_data["detection_confidence"]

                pose_score = 1.0

                quality_score = self.quality_service.calculate_quality_score(blur_score, detection_confidence, pose_score)

                is_valid = self.quality_service.is_valid_frame(blur_score, detection_confidence, quality_score)

                print(f"{image_name} | "), f"quality={quality_score:.4f}"

                if not is_valid:
                    print(f"Rejected: {image_name}")
                    continue

                valid_frame_count += 1

                face_embedding = FaceEmbedding(
                    employee_id=employee_db_id,
                    image_id=face_image.id,
                    embedding=face_data["embedding"],
                    pose_type="UNKNOWN",
                    quality_score=quality_score,
                    blur_score=blur_score,
                    detection_confidence=detection_confidence,
                    yaw_angle=0.0,
                    model_name="arcface"
                )
                
                db.add(face_embedding)

                embedding_count += 1

            db.commit()

            # ==============================
            # Audit Log
            # ==============================

            log = AuditLog(
                action = "EMPLOYEE_ENROLLED",
                description=(f"{employee.employee_db_id} enrolled with {embedding_count} embeddings")
            )

            db.add(log)
            db.commit()

            print(f"Enrollment Completed ({embedding_count} embeddings)")

            return {
                "success": True,
                "employee_id": employee.employee_id,
                "employee_name": employee.full_name,
                "valid_frames": valid_frame_count,
                "embeddings": embedding_count
            }
        
        except Exception as e:

            db.rollback()
            print("Enrollment Error: ", e)

            return {
                "success": False,
                "Message": str(e)
            }
        
        finally:
            db.close()