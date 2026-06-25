from sqlalchemy import (Column, Integer, String, DateTime, Date, Time, ForeignKey, Text, Boolean, JSON, Float)
from sqlalchemy.orm import relationship
from datetime import datetime

from database.database import Base


# ==========================================
# USERS TABLE
# ==========================================
class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)

    employee_id = Column(
        String(50),
        unique=True,
        nullable=False
    )

    full_name = Column(
        String(100),
        nullable=False
    )

    department = Column(
        String(100),
        nullable=False
    )

    designation = Column(
        String(100),
        nullable=True
    )

    email = Column(
        String(100),
        nullable=True
    )

    is_active = Column(
        Boolean,
        default=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# =========================================
# FACE IMAGES
# =========================================
class FaceImage(Base):
    __tablename__ = "face_images"

    id = Column(Integer, primary_key=True)

    employee_id = Column(
        Integer,
        ForeignKey("employees.id")
    )

    image_path = Column(
        String(255),
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# =========================================
# FACE EMBEDDINGS
# =========================================
class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True)

    employee_id = Column(
        Integer,
        ForeignKey("employees.id")
    )

    image_id = Column(
        Integer,
        ForeignKey("face_images.id")
    )

    embedding = Column(JSON)

    pose_type = Column(
        String(20)
    )

    quality_score = Column(
        Float,
        default=0.0
    )

    blur_score = Column(
        Float,
        default=0.0
    )

    detection_confidence = Column(
        Float,
        default=0.0
    )

    yaw_angle = Column(
        Float,
        default=0.0
    )

    model_name = Column(
        String(50),
        default="arcface"
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# ==========================================
# ATTENDANCE TABLE
# ==========================================
class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True)

    employee_id = Column(
        Integer,
        ForeignKey("employees.id")
    )

    attendance_date = Column(
        Date,
        nullable=False
    )

    check_in = Column(Time)

    check_out = Column(Time)

    status = Column(
        String(20),
        default="Present"
    )


# ==========================================
# SECURITY EVENTS
# ==========================================
class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True)

    event_type = Column(
        String(100),
        nullable=False
    )

    employee_id = Column(
        Integer,
        ForeignKey("employees.id"),
        nullable=True
    )

    image_path = Column(
        String(255)
    )

    description = Column(Text)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# ==========================================
# AUDIT LOGS
# ==========================================
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)

    action = Column(
        String(255),
        nullable=False
    )

    details = Column(Text)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )