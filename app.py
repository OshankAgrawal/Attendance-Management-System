from flask import Flask
from routes.admin_routes import admin_bp
from routes.employee_routes import employee_bp
from routes.attendance_routes import attendance_bp

app = Flask(__name__)

app.register_blueprint(admin_bp)
app.register_blueprint(employee_bp)
app.register_blueprint(attendance_bp)

# @app.route("/")
# def dashboard():
#     return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)