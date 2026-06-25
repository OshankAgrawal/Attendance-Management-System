# AI-Powered Biometric Attendance Management System

## Introduction

The AI-Powered Biometric Attendance Management System is an intelligent attendance tracking solution that leverages Face Recognition, Liveness Detection, Challenge-Response Verification, and Quality-Based Embedding Management to ensure secure and accurate employee attendance.

Unlike traditional attendance systems, this application incorporates advanced Artificial Intelligence techniques to prevent spoofing attacks, reduce false positives, and improve recognition accuracy through weighted embedding matching.

The system provides a complete workflow from employee registration and face enrollment to attendance verification, reporting, and security event monitoring through an interactive Flask-based web dashboard.

---

## Features

### Employee Management

* Register new employees
* Store employee details in database
* Manage employee records
* View employee directory

### Face Enrollment

* Capture multiple facial samples using webcam
* Generate ArcFace embeddings
* Store high-quality embeddings only
* Automatic quality filtering
* Pose-aware enrollment support
* Blur detection and confidence scoring

### AI-Powered Face Verification

* ArcFace-based face recognition
* Embedding similarity matching
* Weighted average similarity calculation
* Multi-sample comparison
* Improved recognition accuracy
* Threshold-based verification

### Liveness Detection

* Blink detection using MediaPipe Face Mesh
* Head movement analysis
* Anti-spoofing protection
* Challenge-response verification

### Challenge Verification

Randomized security challenges:

* Blink Twice
* Turn Left
* Turn Right

The user must successfully complete the generated challenge before attendance can be marked.

### Attendance Management

* Check-In functionality
* Check-Out functionality
* Attendance history tracking
* Duplicate attendance prevention
* Daily attendance records

### Security Monitoring

* Liveness failure logging
* No-face detection logging
* Identity verification failure logging
* Security incident tracking
* Audit trail maintenance

### Reporting Dashboard

* Employee statistics
* Attendance statistics
* Security event reports
* Attendance records overview
* Security event summary

---

# System Architecture

## Three-Layer Architecture

### 1. AI Layer

Responsible for:

* Face Detection
* Face Alignment
* Embedding Generation
* Quality Analysis
* Liveness Detection
* Challenge Verification
* Face Recognition

### 2. Business Logic Layer

Responsible for:

* Employee Management
* Enrollment Workflow
* Attendance Workflow
* Security Event Management
* Reporting

### 3. Presentation Layer

Built using:

* Flask
* HTML
* CSS
* Jinja Templates

Provides:

* Dashboard
* Employee Management UI
* Enrollment UI
* Attendance UI
* Reports UI
* Security Events UI

---

# Technology Stack

## Backend

* Python
* Flask
* SQLAlchemy

## Database

* MySQL
* phpMyAdmin

## AI & Computer Vision

* OpenCV
* InsightFace
* ArcFace
* MediaPipe
* NumPy

## Frontend

* HTML5
* CSS3
* Jinja2 Templates

---

# Project Structure

```text
Attendance-Management-System/

│
├── database/
│   ├── database.py
│   └── models.py
│
├── routes/
│   ├── admin_routes.py
|   ├── attendance_route.py
|   ├── employee_routes.py
|   ├── auth_route.py
│
├── services/
│   ├── attendance_service.py
│   ├── attendance_verification_service.py
│   ├── embedding_service.py
│   ├── enrollment_service.py
│   ├── liveness_service.py
│   ├── security_event_service.py
│   └── verification_service.py
│
├── static/
│   └── css/
│
├── templates/
|   ├── attendance
|   |   ├── attendance_result.html
|   |   └── attendance.html
|   |
│   ├── base.html
│   ├── dashboard.html
│   ├── employees.html
│   ├── register_employee.html
│   ├── enroll_employee.html
│   ├── security_events.html
│   └── reports.html
│
├── app.py
│
├── requirements.txt
|
├── LICENSE
│
└── README.md
```

---

# Database Design

## Employees

Stores employee information.

### Fields

* id
* employee_id
* full_name
* department
* designation
* status

---

## Face Images

Stores enrollment images.

### Fields

* id
* employee_id
* image_path
* pose_type

---

## Face Embeddings

Stores ArcFace embeddings and quality metadata.

### Fields

* id
* employee_id
* image_id
* embedding
* pose_type
* quality_score
* blur_score
* detection_confidence
* yaw_angle
* model_name

---

## Attendance

Stores attendance records.

### Fields

* id
* employee_id
* attendance_date
* check_in
* check_out
* status

---

## Security Events

Stores security incidents.

### Fields

* id
* event_type
* employee_id
* image_path
* description
* created_at

---

# Enrollment Workflow

```text
Register Employee
        ↓
Open Webcam
        ↓
Capture Multiple Frames
        ↓
Quality Analysis
        ↓
Generate ArcFace Embeddings
        ↓
Store Valid Embeddings
        ↓
Enrollment Complete
```

---

# Attendance Verification Workflow

```text
Start Attendance
        ↓
Generate Challenge
        ↓
Challenge Verification
        ↓
Liveness Detection
        ↓
Face Recognition
        ↓
Weighted Similarity Calculation
        ↓
Attendance Marked
```

---

# Weighted Similarity Matching

The system uses a weighted average similarity approach.

During enrollment:

* Low-quality samples are discarded.
* Only high-quality embeddings are stored.

During verification:

* Similarity is calculated against all stored embeddings.
* Weighted averaging is applied using quality scores.
* Final similarity score is compared against verification threshold.

This approach improves recognition reliability and reduces false matches.

---

# Security Features

### Anti-Spoofing

* Blink Detection
* Head Movement Validation
* Challenge-Response Verification

### Security Logging

Automatically logs:

* Liveness Failures
* No Face Detected
* Identity Verification Failures
* Unknown Person Attempts

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Attendance-Management-System.git

cd Attendance-Management-System
```

## Create Virtual Environment

```bash
python -m venv myenv
```

### Activate Environment

Windows

```bash
myenv\Scripts\activate
```

Linux / Mac

```bash
source myenv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Database Setup

Create MySQL database:

```sql
CREATE DATABASE biometric_attendance;
```

Update database configuration in:

```python
database/database.py
```

Example:

```python
DATABASE_URL = "mysql+pymysql://root:password@localhost/biometric_attendance"
```

---

# Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# Future Enhancements

* User Authentication
* Role-Based Access Control
* Browser-Based Camera Integration
* Attendance Export (Excel/PDF)
* Email Notifications
* Real-Time Dashboard Analytics
* Cloud Deployment
* Mobile Application Support
* Multi-Face Attendance Support

---

## License & Usage Notice

This project is licensed under the **MIT License**.

You are free to:

* Use this project for educational purposes
* Modify and extend the source code
* Share and distribute the project with proper attribution

Please provide appropriate credit to the original author when reusing or modifying this work.

---

# Author

**Oshank Agrawal**
B.Tech – Artificial Intelligence & Data Science
Samrat Ashok Technological Institute, Vidisha

### Connect With Me

* LinkedIn: *[OshankAgrawal](https://www.linkedin.com/in/oshankagrawal/)*
* Email: *[oshankagrawal](mailto:oshankagrawal@gmail.com)*
* GitHub: *[OshankAgrawal](https://github.com/OshankAgrawal)*

---

# Project Highlights

This project demonstrates the practical implementation of:

* Computer Vision
* Face Recognition using ArcFace
* Liveness Detection
* Challenge-Response Verification
* AI-Powered Attendance Management
* Flask Web Development
* Database Design & Integration
* Security Event Monitoring

The system was designed with a focus on security, accuracy, and real-world usability, combining modern AI techniques with enterprise-style attendance management workflows.

---

# Future Scope

Potential enhancements for future versions include:

* User Authentication & Role-Based Access Control
* Browser-Based Camera Integration
* Attendance Export (Excel/PDF)
* Email Notifications
* Cloud Deployment
* Real-Time Analytics Dashboard
* Mobile Application Support
* Multi-Camera Attendance Verification

---

# Acknowledgements

Special thanks to the open-source community and the developers behind:

* OpenCV
* InsightFace
* ArcFace
* MediaPipe
* Flask
* SQLAlchemy
* NumPy

whose tools and libraries made this project possible.

---

# Final Note

This project represents the integration of Artificial Intelligence, Computer Vision, and Web Development to solve a real-world problem in attendance management.

Beyond simply marking attendance, the system focuses on identity verification, anti-spoofing protection, security monitoring, and reliable recognition, making it significantly more robust than traditional attendance solutions.

⭐ If you found this project useful, consider giving it a star on GitHub.
