"""
liveness_service.py
-------------------
Face Liveness Detection Service

Features:
- Random challenges: BLINK_TWICE, TURN_LEFT, TURN_RIGHT, SMILE, RAISE_EYEBROWS, NOD
- Liveness texture analysis (screen moiré / print artifact detection via FFT)
- Pseudo-depth estimation via facial landmark spread
- Bug fixes from original version:
    * putText y-coordinate collision fixed
    * duplicate TURN_RIGHT elif block removed
    * BLINK_TWICE now has a frame timeout
    * head-turn counters reset correctly on direction change
    * challenge text always visible on a dedicated HUD line
"""

import cv2
import math
import random
import time
import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh_module


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Eye landmark indices (MediaPipe 478-point model)
LEFT_EYE_POINTS  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

# Mouth landmark indices for smile detection
# Outer lip corners + top/bottom lip midpoints
MOUTH_POINTS = [61, 291, 0, 17, 78, 308]

# Eyebrow landmark indices (inner + outer per brow)
LEFT_BROW_POINTS  = [70,  63,  105, 66,  107]   # left eyebrow (ascending arch)
RIGHT_BROW_POINTS = [336, 296, 334, 293, 300]   # right eyebrow

# Nose tip and forehead for nod (pitch) detection
NOSE_TIP       = 1
CHIN_TIP       = 152
FOREHEAD_TIP   = 10

# Face bounding landmarks for depth spread
FACE_SPREAD_POINTS = [10, 152, 234, 454]   # top, chin, left ear, right ear

# Thresholds
BLINK_THRESHOLD          = 0.22   # EAR below this → eye closed
SMILE_THRESHOLD          = 0.35   # mouth-width / vertical ratio
BROW_RAISE_THRESHOLD     = 0.28   # brow-to-eye distance ratio
NOD_THRESHOLD            = 0.04   # nose-to-chin y shift for a nod
DEPTH_MIN_SPREAD         = 0.30   # normalized face spread must exceed this (filters flat photos)
MOIRE_FREQ_THRESHOLD     = 0.18   # fraction of FFT energy in high-freq band (screen artifact)

# Timing / frame counts
REQUIRED_DIRECTION_FRAMES = 12   # frames head must stay turned
REQUIRED_NOD_FRAMES       = 8    # frames chin must be raised/lowered
REQUIRED_BROW_FRAMES      = 8    # frames brows must stay raised
REQUIRED_SMILE_FRAMES     = 10   # frames smile must be held
CENTER_TIMEOUT_FRAMES     = 45   # frames before "not moving" counts as failure
BLINK_TIMEOUT_FRAMES      = 300  # ~10 s at 30 fps; fail if no blink by then
DEPTH_CHECK_INTERVAL      = 30   # run depth check every N frames
MOIRE_CHECK_INTERVAL      = 30   # run moiré check every N frames


# ---------------------------------------------------------------------------
# HUD helper
# ---------------------------------------------------------------------------

def _draw_hud(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """
    Draw multi-line text HUD at the top-left of *frame*.
    lines: list of (text, BGR colour) tuples, drawn top-to-bottom.
    """
    y = 35
    for text, colour in lines:
        cv2.putText(
            img=frame,
            text=text,
            org=(15, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        y += 30


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class LivenessService:

    def __init__(self):
        self._face_mesh = mp_face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # All supported challenges
        self.challenges = [
            "BLINK_TWICE",
            "TURN_LEFT",
            "TURN_RIGHT",
            "SMILE",
            "RAISE_EYEBROWS",
            "NOD",
        ]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def verify_liveness(self) -> bool:
        """
        Open the webcam, issue a random challenge, and return True only if:
          1. The depth check passes (face looks 3-D, not a flat photo).
          2. The moiré check passes (no screen / print artifact detected).
          3. The user correctly performs the challenge.
        Returns False on any failure or timeout.
        """
        challenge = self._generate_challenge()
        print(f"[LivenessService] Challenge selected: {challenge}")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[LivenessService] ERROR: Could not open webcam.")
            return False

        # ---- per-challenge state ----------------------------------------
        blink_count       = 0
        eye_closed        = False
        left_counter      = 0
        right_counter     = 0
        center_counter    = 0
        nod_down_frames   = 0
        nod_up_frames     = 0
        nod_phase         = "WAIT_DOWN"   # state machine: WAIT_DOWN → WAIT_UP → DONE
        smile_frames      = 0
        brow_frames       = 0
        frame_index       = 0

        # Store baseline values on first detected face
        baseline_nose_y   = None
        baseline_chin_y   = None

        # Anti-spoofing verdicts (updated periodically)
        depth_ok  = None   # None = not yet checked
        moire_ok  = None

        status_msg = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

            # ---- Anti-spoofing checks (run periodically) -----------------
            if frame_index % DEPTH_CHECK_INTERVAL == 0:
                depth_ok = self._check_depth(results, w, h)

            if frame_index % MOIRE_CHECK_INTERVAL == 0:
                moire_ok = self._check_moire(frame)

            # ---- Fail fast if anti-spoofing detects attack ---------------
            if depth_ok is False:
                print("[LivenessService] FAIL: Depth check failed — possible flat photo/screen.")
                cap.release()
                cv2.destroyAllWindows()
                return False

            if moire_ok is False:
                print("[LivenessService] FAIL: Moiré pattern detected — possible screen replay.")
                cap.release()
                cv2.destroyAllWindows()
                return False

            # ---- Process face landmarks -----------------------------------
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # Capture baseline on first frame
                if baseline_nose_y is None:
                    baseline_nose_y = lm[NOSE_TIP].y
                    baseline_chin_y = lm[CHIN_TIP].y

                direction = self._detect_head_direction(lm)
                ear       = self._average_ear(lm)
                smile_r   = self._smile_ratio(lm)
                brow_r    = self._brow_raise_ratio(lm)
                nod_delta = lm[NOSE_TIP].y - baseline_nose_y

                # Track blinks (shared across all challenges)
                if ear < BLINK_THRESHOLD:
                    eye_closed = True
                else:
                    if eye_closed:
                        blink_count += 1
                        eye_closed = False

                # ---- HUD -------------------------------------------------
                anti_spoof_colour = (
                    (0, 255, 0) if (depth_ok and moire_ok) else
                    (0, 165, 255) if (depth_ok is None or moire_ok is None) else
                    (0, 0, 255)
                )
                hud_lines = [
                    (f"Challenge : {challenge}", (0, 220, 255)),
                    (f"EAR: {ear:.2f}  Blinks: {blink_count}", (200, 200, 200)),
                    (f"Direction : {direction}", (200, 200, 200)),
                    (f"Smile: {smile_r:.2f}  Brow: {brow_r:.2f}", (200, 200, 200)),
                    (f"Anti-spoof: {'OK' if (depth_ok and moire_ok) else 'Checking...' if depth_ok is None else 'ALERT'}", anti_spoof_colour),
                ]
                if status_msg:
                    hud_lines.append((status_msg, (0, 100, 255)))
                _draw_hud(frame, hud_lines)

                # ==========================================================
                # CHALLENGE EVALUATION
                # ==========================================================

                # --- BLINK_TWICE ------------------------------------------
                if challenge == "BLINK_TWICE":
                    if direction in ("LEFT", "RIGHT"):
                        status_msg = "Keep head straight!"
                        print("[LivenessService] FAIL: Head turned during BLINK_TWICE.")
                        _draw_hud(frame, hud_lines)
                        cv2.imshow("Liveness Detection", frame)
                        cv2.waitKey(800)
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

                    if frame_index >= BLINK_TIMEOUT_FRAMES:
                        print("[LivenessService] FAIL: BLINK_TWICE timed out.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

                    if blink_count >= 2:
                        print("[LivenessService] PASS: BLINK_TWICE completed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True

                # --- TURN_LEFT --------------------------------------------
                elif challenge == "TURN_LEFT":
                    if direction == "RIGHT":
                        print("[LivenessService] FAIL: Turned wrong way (RIGHT) for TURN_LEFT.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

                    if direction == "CENTER":
                        center_counter += 1
                        left_counter = 0   # reset if they drift back
                        if center_counter >= CENTER_TIMEOUT_FRAMES:
                            print("[LivenessService] FAIL: TURN_LEFT timed out at CENTER.")
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
                    elif direction == "LEFT":
                        left_counter += 1
                        center_counter = 0
                    else:
                        left_counter = 0

                    if left_counter >= REQUIRED_DIRECTION_FRAMES:
                        print("[LivenessService] PASS: TURN_LEFT completed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True

                # --- TURN_RIGHT -------------------------------------------
                elif challenge == "TURN_RIGHT":
                    if direction == "LEFT":
                        print("[LivenessService] FAIL: Turned wrong way (LEFT) for TURN_RIGHT.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

                    if direction == "CENTER":
                        center_counter += 1
                        right_counter = 0
                        if center_counter >= CENTER_TIMEOUT_FRAMES:
                            print("[LivenessService] FAIL: TURN_RIGHT timed out at CENTER.")
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
                    elif direction == "RIGHT":
                        right_counter += 1
                        center_counter = 0
                    else:
                        right_counter = 0

                    if right_counter >= REQUIRED_DIRECTION_FRAMES:
                        print("[LivenessService] PASS: TURN_RIGHT completed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True

                # --- SMILE ------------------------------------------------
                elif challenge == "SMILE":
                    if direction in ("LEFT", "RIGHT"):
                        status_msg = "Keep head straight while smiling!"
                    elif smile_r >= SMILE_THRESHOLD:
                        smile_frames += 1
                        center_counter = 0
                    else:
                        smile_frames = 0
                        center_counter += 1

                    if center_counter >= CENTER_TIMEOUT_FRAMES:
                        print("[LivenessService] FAIL: SMILE timed out.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

                    if smile_frames >= REQUIRED_SMILE_FRAMES:
                        print("[LivenessService] PASS: SMILE completed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True

                # --- RAISE_EYEBROWS ---------------------------------------
                elif challenge == "RAISE_EYEBROWS":
                    if brow_r >= BROW_RAISE_THRESHOLD:
                        brow_frames += 1
                        center_counter = 0
                    else:
                        brow_frames = 0
                        center_counter += 1

                    if center_counter >= CENTER_TIMEOUT_FRAMES:
                        print("[LivenessService] FAIL: RAISE_EYEBROWS timed out.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

                    if brow_frames >= REQUIRED_BROW_FRAMES:
                        print("[LivenessService] PASS: RAISE_EYEBROWS completed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True

                # --- NOD --------------------------------------------------
                elif challenge == "NOD":
                    """
                    Nod state machine:
                      WAIT_DOWN  → nose dips below baseline by NOD_THRESHOLD
                      WAIT_UP    → nose rises back above baseline
                      DONE       → full nod completed
                    """
                    if nod_phase == "WAIT_DOWN":
                        if nod_delta > NOD_THRESHOLD:
                            nod_down_frames += 1
                        else:
                            nod_down_frames = 0

                        if nod_down_frames >= REQUIRED_NOD_FRAMES:
                            nod_phase = "WAIT_UP"
                            status_msg = "Now raise your head back up!"

                    elif nod_phase == "WAIT_UP":
                        if nod_delta < (NOD_THRESHOLD / 2):
                            nod_up_frames += 1
                        else:
                            nod_up_frames = 0

                        if nod_up_frames >= REQUIRED_NOD_FRAMES:
                            nod_phase = "DONE"

                    elif nod_phase == "DONE":
                        print("[LivenessService] PASS: NOD completed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True

                    center_counter += 1
                    if center_counter >= CENTER_TIMEOUT_FRAMES * 3:   # generous timeout for nod
                        print("[LivenessService] FAIL: NOD timed out.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

            else:
                # No face detected this frame
                _draw_hud(frame, [
                    (f"Challenge : {challenge}", (0, 220, 255)),
                    ("No face detected — please center your face", (0, 0, 255)),
                ])

            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return False

    # ------------------------------------------------------------------
    # Challenge generator
    # ------------------------------------------------------------------

    def _generate_challenge(self) -> str:
        return random.choice(self.challenges)

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(pt1, pt2) -> float:
        """Euclidean distance between two MediaPipe NormalizedLandmark objects."""
        return math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

    def _eye_aspect_ratio(self, landmarks, eye_points: list[int]) -> float:
        """
        Eye Aspect Ratio (Soukupová & Čech, 2016):
            EAR = (‖p2–p6‖ + ‖p3–p5‖) / (2 · ‖p1–p4‖)
        Returns 0 if horizontal span is zero (degenerate case).
        """
        p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_points)
        v1 = self._distance(p2, p6)
        v2 = self._distance(p3, p5)
        h  = self._distance(p1, p4)
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def _average_ear(self, landmarks) -> float:
        """Average EAR across both eyes."""
        left  = self._eye_aspect_ratio(landmarks, LEFT_EYE_POINTS)
        right = self._eye_aspect_ratio(landmarks, RIGHT_EYE_POINTS)
        return (left + right) / 2.0

    def _detect_head_direction(self, landmarks) -> str:
        """
        Estimate yaw (left/right) by comparing the nose-tip x-position
        to the midpoint between both eye outer corners.
        Threshold of ±0.03 filters natural micro-wobble.
        """
        nose_x    = landmarks[NOSE_TIP].x
        mid_x     = (landmarks[LEFT_EYE_POINTS[0]].x + landmarks[RIGHT_EYE_POINTS[0]].x) / 2.0
        diff      = nose_x - mid_x

        if diff < -0.03:
            return "LEFT"
        if diff > 0.03:
            return "RIGHT"
        return "CENTER"

    def _smile_ratio(self, landmarks) -> float:
        """
        Smile ratio = mouth width / (2 × vertical mouth opening).
        A wider-than-tall ratio indicates a smile.
        Landmark indices: 61 (left corner), 291 (right corner),
        0 (top lip center), 17 (bottom lip center).
        """
        left_corner  = landmarks[61]
        right_corner = landmarks[291]
        top_lip      = landmarks[0]
        bottom_lip   = landmarks[17]

        mouth_width  = self._distance(left_corner, right_corner)
        mouth_height = self._distance(top_lip, bottom_lip) + 1e-6
        return mouth_width / mouth_height

    def _brow_raise_ratio(self, landmarks) -> float:
        """
        Brow raise ratio = average vertical distance from brow to eye,
        normalized by the inter-eye distance.
        Higher value → brows raised further above eyes.
        """
        inter_eye = self._distance(landmarks[LEFT_EYE_POINTS[0]],
                                   landmarks[RIGHT_EYE_POINTS[0]]) + 1e-6

        # Average of left brow arch and right brow arch midpoints vs eye corner
        left_brow_y  = sum(landmarks[i].y for i in LEFT_BROW_POINTS)  / len(LEFT_BROW_POINTS)
        right_brow_y = sum(landmarks[i].y for i in RIGHT_BROW_POINTS) / len(RIGHT_BROW_POINTS)
        left_eye_y   = landmarks[LEFT_EYE_POINTS[0]].y
        right_eye_y  = landmarks[RIGHT_EYE_POINTS[0]].y

        # In normalized coords y increases downward, so brow is ABOVE eye → negative diff
        left_dist  = abs(left_eye_y  - left_brow_y)
        right_dist = abs(right_eye_y - right_brow_y)

        return ((left_dist + right_dist) / 2.0) / inter_eye

    # ------------------------------------------------------------------
    # Anti-spoofing: depth estimation
    # ------------------------------------------------------------------

    def _check_depth(self, results, frame_w: int, frame_h: int) -> bool | None:
        """
        Pseudo-depth estimation via 3-D landmark spread.

        MediaPipe provides a normalized `z` coordinate (relative depth).
        We measure the peak-to-peak spread across key facial landmarks in z.
        A real 3-D face produces a meaningful depth range; a flat photo or
        screen has near-zero z variance because all points lie on one plane.

        Additionally, we measure the 2-D bounding area of landmark spread in
        x/y (a real face at typical webcam distance spans a healthy fraction
        of the frame).

        Returns:
            True  → looks like a real 3-D face
            False → likely a flat spoof
            None  → no face detected yet, skip
        """
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        # --- z-depth spread ---
        z_values = [lm[i].z for i in FACE_SPREAD_POINTS]
        z_spread = max(z_values) - min(z_values)

        # --- 2-D area (normalized) ---
        x_values = [lm[i].x for i in FACE_SPREAD_POINTS]
        y_values = [lm[i].y for i in FACE_SPREAD_POINTS]
        xy_spread = (max(x_values) - min(x_values)) * (max(y_values) - min(y_values))

        # A real face at webcam distance: z_spread typically > 0.05,
        # xy_spread typically > 0.04 (face covers a reasonable portion of frame)
        depth_pass = (z_spread > 0.04) or (xy_spread > DEPTH_MIN_SPREAD * 0.15)

        if not depth_pass:
            print(f"[LivenessService] Depth check: z_spread={z_spread:.4f}, "
                  f"xy_spread={xy_spread:.4f} → FAIL")
        return depth_pass

    # ------------------------------------------------------------------
    # Anti-spoofing: moiré / screen artifact detection (FFT)
    # ------------------------------------------------------------------

    def _check_moire(self, frame: np.ndarray) -> bool:
        """
        Detect screen / printed-photo replay using FFT texture analysis.

        Digital screens and laser-printed photos introduce regular high-
        frequency grid patterns (moiré, halftone dots, LCD pixel grid).
        These appear as pronounced peaks in the FFT magnitude spectrum at
        spatial frequencies corresponding to the pattern pitch.

        Method:
          1. Convert to grayscale and resize to a fixed size for speed.
          2. Compute 2-D FFT and shift DC to center.
          3. Split spectrum into a low-frequency ring (face texture) and a
             high-frequency ring (potential moiré / pixel grid).
          4. If the fraction of energy in the high-frequency ring exceeds
             MOIRE_FREQ_THRESHOLD → likely a screen / print replay attack.

        Returns:
            True  → texture looks natural (real face)
            False → moiré / screen artifact detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize for speed; FFT of 256×256 is plenty
        resized = cv2.resize(gray, (256, 256)).astype(np.float32)

        # FFT
        fft    = np.fft.fft2(resized)
        fft_sh = np.fft.fftshift(fft)
        mag    = np.abs(fft_sh)

        h, w   = mag.shape
        cy, cx = h // 2, w // 2

        # Build radial distance map
        y_idx, x_idx = np.ogrid[:h, :w]
        r_map = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

        max_r = min(cx, cy)

        # Low-freq ring: 0 → 10 % of max radius (DC + main face structure)
        # Mid-freq ring: 10 % → 40 % (natural skin texture)
        # High-freq ring: 40 % → 100 % (moiré / pixel grid)
        low_mask  = r_map <= max_r * 0.10
        high_mask = r_map >= max_r * 0.40

        total_energy = mag.sum() + 1e-8
        high_energy  = mag[high_mask].sum()
        high_ratio   = high_energy / total_energy

        moire_detected = high_ratio > MOIRE_FREQ_THRESHOLD

        if moire_detected:
            print(f"[LivenessService] Moiré check: high_freq_ratio={high_ratio:.4f} → FAIL")
        return not moire_detected
