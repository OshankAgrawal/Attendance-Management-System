from flask import Blueprint, render_template
from database.database import SessionLocal
from database.models import Employee, Attendance, SecurityEvent
from datetime import date
from sqlalchemy import func

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/")
def home():
    return dashboard()

@admin_bp.route("/dashboard")
def dashboard():

    db = SessionLocal()

    total_employees = db.query(Employee).count()

    present_today = db.query(Attendance).filter(Attendance.attendance_date == date.today()).count()

    absent_today = total_employees - present_today

    security_events = db.query(SecurityEvent).count()

    db.close()

    return render_template(
        "dashboard.html",
        total_employees=total_employees,
        present_today=present_today,
        absent_today=absent_today,
        security_events=security_events
    )

@admin_bp.route("/security-events")
def security_events():

    db = SessionLocal()

    try:
        events = db.query(SecurityEvent).order_by(SecurityEvent.created_at.desc()).all()

        return render_template("security_events.html", events=events)
    
    except Exception as e:
        raise(e)
        
    finally:
        db.close()

@admin_bp.route("/reports")
def reports():

    db = SessionLocal()

    try:
        total_employees = db.query(Employee).count()

        present_today = db.query(Attendance).filter(Attendance.attendance_date == date.today()).count()

        absent_today = total_employees - present_today

        total_security_events = db.query(SecurityEvent).count()

        attendance_records = (
            db.query(
                Attendance,
                Employee
            )
            .join(
                Employee,
                Attendance.employee_id == Employee.id
            )
            .order_by(
                Attendance.attendance_date.desc()
            )
            .all()
        )

        security_summary = (
            db.query(
                SecurityEvent.event_type,
                func.count(SecurityEvent.id)
            )
            .group_by(
                SecurityEvent.event_type
            )
            .all()
        )

        return render_template(
            "reports.html",

            total_employees=total_employees,
            present_today=present_today,
            absent_today=absent_today,
            total_security_events=total_security_events,

            attendance_records=attendance_records,
            security_summary=security_summary
        )
    finally:
        db.close()