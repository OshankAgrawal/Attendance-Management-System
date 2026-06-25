from flask import Blueprint, render_template, request, redirect, url_for
from database.database import SessionLocal
from database.models import Employee

employee_bp = Blueprint("employee", __name__)

@employee_bp.route("/employees")
def employees():

    db = SessionLocal()

    employees = db.query(Employee).all()

    db.close()

    return render_template(
        "employees.html",
        employees=employees
    )

@employee_bp.route("/register", methods=["GET", "POST"])
def register_employee():

    if request.method == "POST":

        db = SessionLocal()

        employee = Employee(
            employee_id = request.form["employee_id"],
            full_name=request.form["full_name"],
            department=request.form["department"],
            designation=request.form["designation"],
            email=request.form["email"],
            is_active=True
        )

        db.add(employee)
        db.commit()

        employee_id = employee.id

        db.close()

        return redirect(url_for("employee.enrollment_page", employee_id=employee_id))
    
    return render_template("register_employee.html")

@employee_bp.route("/enroll/<int:employee_id>")
def enrollment_page(employee_id):

    db = SessionLocal()

    employee = db.query(Employee).filter(Employee.id == employee_id).first()

    db.close()

    return render_template("enroll_employee.html", employee=employee)

@employee_bp.route("/start-enrollment/<int:employee_id>")
def start_enrollment(employee_id):

    from services.enrollment_service import EnrollmentService

    enrollment_service = EnrollmentService()

    result = enrollment_service.start_face_enrollment(employee_id)

    return render_template("enrollment_complete.html", result=result)