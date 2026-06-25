import cv2
from insightface.app import FaceAnalysis

from services.security_event_service import SecurityEventService

class EmbeddingService:

    def __init__(self):

        self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.security_logger = SecurityEventService()

        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def generate_embedding(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot load {image_path}")

        faces = self.app.get(image)

        print(f"Faces Detected: {len(faces)}")

        if len(faces) == 0:
            return None
        
        if len(faces) > 1:
            self.security_logger.log_event(event_type="MULTIPLE_FACES")

        face = faces[0]

        print("Bounding Box:", face.bbox)

        embedding = face.embedding

        return embedding.tolist()
    
    def generate_face_data(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Unable to load {image_path}")
        
        faces = self.app.get(image)

        if len(faces) == 0:
            return None
        
        face = faces[0]

        return {
            "embedding": face.embedding.tolist(),
            "detection_confidence": float(face.det_score),
            "bbox": face.bbox.tolist(),
            "landmarks": face.kps.tolist()
        }