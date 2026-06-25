from flask import Blueprint, render_template

attendance_bp = Blueprint("attendance", __name__)

@attendance_bp.route("/attendance")
def attendance_page():

    return render_template("attendance/attendance.html")

@attendance_bp.route("/start-attendance")
def start_attendance():

    from services.attendance_verification_service import AttendanceVerificationService

    service = AttendanceVerificationService()

    result = service.process_attendance_from_webcam()

    return render_template("attendance/attendance_result.html", result=result)