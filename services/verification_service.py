import cv2
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from database.database import SessionLocal
from database.models import Employee, FaceEmbedding

from services.embedding_service import EmbeddingService
from services.security_event_service import SecurityEventService

class VerificationService:

    def __init__(self):

        self.embedding_service = EmbeddingService()
        self.security_logger = SecurityEventService()

        self.TOP_K = 3

        self.MIN_BEST_SCORE = 0.85

        self.MIN_WEIGHTED_AVG = 0.82

    def verify_image(self, image_path):

        db = SessionLocal()

        try:

            live_embedding = self.embedding_service.generate_embedding(image_path)

            if live_embedding is None:

                return {
                    "verified": False,
                    "message": "No face detected"
                }

            live_embedding = np.array(live_embedding).reshape(1, -1)

            employees = (
                db.query(Employee)
                .filter(
                    Employee.is_active == True
                )
                .all()
            )

            best_employee = None

            best_similarity = 0.0

            best_topk_avg = 0.0

            best_weighted_avg = 0.0

            for employee in employees:

                embeddings = (
                    db.query(FaceEmbedding)
                    .filter(
                        FaceEmbedding.employee_id
                        == employee.id
                    )
                    .all()
                )

                if not embeddings:
                    continue

                scores = []

                weighted_sum = 0.0

                quality_sum = 0.0

                for emb in embeddings:

                    stored_embedding = np.array(emb.embedding).reshape(1, -1)

                    similarity = cosine_similarity(
                        live_embedding,
                        stored_embedding
                    )[0][0]

                    quality = emb.quality_score

                    scores.append(float(similarity))

                    weighted_sum += similarity * quality

                    quality_sum += quality

                scores.sort(reverse=True)

                top_scores = scores[:self.TOP_K]

                topk_avg = sum(top_scores) / len(top_scores)

                if quality_sum > 0:
                    weighted_avg = weighted_sum / quality_sum
                else:
                    weighted_avg = 0.0

                employee_best = max(scores)

                if weighted_avg > best_weighted_avg:
                    best_weighted_avg = weighted_avg
                    best_topk_avg = topk_avg
                    best_similarity = employee_best
                    best_employee = employee

            if (best_employee
                and
                best_similarity >= self.MIN_BEST_SCORE
                and
                best_weighted_avg >= self.MIN_WEIGHTED_AVG
            ):

                return {

                    "verified": True,

                    "employee_id": best_employee.employee_id,

                    "employee_name": best_employee.full_name,

                    "best_similarity": round(best_similarity, 4),

                    "weighted_average": round(best_weighted_avg, 4),

                    "top3_average": round(best_topk_avg, 4)
                }
            
            self.security_logger.log_event(
                event_type="UNKNOWN_FACE",
                description=(f"Best Similarity: ", f"{best_similarity:.4f}")
            )

            self.security_logger.log_event(event_type="NO_FACE_DETECTED")

            return {

                "verified": False,

                "best_similarity": round(best_similarity, 4),

                "top3_average": round(best_topk_avg, 4),

                "message": "Unknown Person"
            }

        finally:

            db.close()