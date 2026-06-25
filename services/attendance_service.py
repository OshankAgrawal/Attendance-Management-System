from datetime import datetime, date

from database.database import SessionLocal
from database.models import Attendance


class AttendanceService:

    def __init__(self):
        pass

    def mark_attendance(self, employee_db_id):

        db = SessionLocal()

        try:

            today = date.today()

            attendance = (
                db.query(Attendance)
                .filter(
                    Attendance.employee_id == employee_db_id,
                    Attendance.attendance_date == today
                )
                .first()
            )

            current_time = (
                datetime.now()
                .time()
                .replace(microsecond=0)
            )

            # -------------------------
            # Check-In
            # -------------------------

            if attendance is None:

                attendance = Attendance(
                    employee_id=employee_db_id,
                    attendance_date=today,
                    check_in=current_time,
                    status="Present"
                )

                db.add(attendance)
                db.commit()

                return {
                    "success": True,
                    "action": "CHECK_IN",
                    "time": str(current_time)
                }

            # -------------------------
            # Check-Out
            # -------------------------

            if attendance.check_out is None:

                attendance.check_out = current_time

                db.commit()

                return {
                    "success": True,
                    "action": "CHECK_OUT",
                    "time": str(current_time)
                }

            # -------------------------
            # Already Completed
            # -------------------------

            return {
                "success": False,
                "message":
                "Attendance already completed today"
            }

        finally:

            db.close()