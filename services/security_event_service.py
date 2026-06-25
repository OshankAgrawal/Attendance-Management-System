from database.database import SessionLocal
from database.models import SecurityEvent

class SecurityEventService:

    def __init__(self):
        pass

    def log_event(self, event_type, description="", employee_id=None):

        db = SessionLocal()

        try:
            event = SecurityEvent(event_type=event_type, employee_id=employee_id, description=description)

            db.add(event)

            db.commit()

            print(f"[SECURITY]", f"{event_type}")

        except Exception as e:
            print(f"Security Log Error: {e}")

        finally:
            db.close()