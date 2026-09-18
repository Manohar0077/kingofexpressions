"""
Wojak Gesture Controller
========================
A real-time hand + face tracking application that switches Wojak meme images
based on detected hand gestures. Inspired by the creative-coding aesthetic of
projects like mundane.zip.

Controls:
    ESC  -> Exit
"""

import sys
import time
import os
import cv2
import numpy as np
import mediapipe as mp

# =============================================================================
# CONSTANTS
# =============================================================================

CAMERA_INDEX   = 0
TARGET_WIDTH   = 1280
TARGET_HEIGHT  = 720

# How many consecutive frames of the same gesture before we commit to it
STABLE_FRAMES  = 5

# Right panel width as a fraction of total window width
MEME_PANEL_RATIO = 0.35

# MediaPipe confidence thresholds
HAND_DETECTION_CONFIDENCE  = 0.7
HAND_TRACKING_CONFIDENCE   = 0.6
FACE_DETECTION_CONFIDENCE  = 0.5
FACE_TRACKING_CONFIDENCE   = 0.5

# Colour palette (BGR)
COLOR_GREEN      = (0,  220,  80)
COLOR_GREEN_DIM  = (0,  100,  40)
COLOR_BLACK      = (0,    0,   0)
COLOR_WHITE      = (255, 255, 255)
COLOR_GRAY       = (80,  80,  80)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Gesture -> asset filename mapping
GESTURE_IMAGES = {
    "NEUTRAL":   "assets/neutral.png",
    "POINTING":  "assets/pointing.png",
    "SOYJAK":    "assets/sojak.png",
    "RODENT":    "assets/middle.png",
    "THUMBS_UP": "assets/thumbs_up.png",
    "PEACE":     "assets/peace.png",
}

# MediaPipe hand landmark indices
WRIST        = 0
THUMB_CMC    = 1
THUMB_MCP    = 2
THUMB_IP     = 3
THUMB_TIP    = 4
INDEX_MCP    = 5
INDEX_PIP    = 6
INDEX_DIP    = 7
INDEX_TIP    = 8
MIDDLE_MCP   = 9
MIDDLE_PIP   = 10
MIDDLE_DIP   = 11
MIDDLE_TIP   = 12
RING_MCP     = 13
RING_PIP     = 14
RING_DIP     = 15
RING_TIP     = 16
PINKY_MCP    = 17
PINKY_PIP    = 18
PINKY_DIP    = 19
PINKY_TIP    = 20


# =============================================================================
# ASSET LOADING
# =============================================================================

def load_assets() -> dict:
    """
    Load all Wojak PNG images from disk into memory once at startup.

    Returns a dict mapping gesture name -> numpy BGRA image (or None if missing).
    Images are loaded with alpha channel (IMREAD_UNCHANGED) so transparency is
    preserved when compositing onto the black panel.
    """
    assets = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for gesture, rel_path in GESTURE_IMAGES.items():
        abs_path = os.path.join(script_dir, rel_path)
        if not os.path.exists(abs_path):
            print(f"  [WARN] Asset not found: {abs_path}  -> will show fallback text for '{gesture}'")
            assets[gesture] = None
        else:
            img = cv2.imread(abs_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  [WARN] Could not decode: {abs_path}  -> will show fallback text for '{gesture}'")
                assets[gesture] = None
            else:
                # Ensure image has an alpha channel
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                assets[gesture] = img
                print(f"  [OK]   Loaded {gesture}: {img.shape[1]}x{img.shape[0]} px")

    return assets


# =============================================================================
# CAMERA INITIALISATION
# =============================================================================

def initialize_camera() -> cv2.VideoCapture:
    """
    Open the default webcam and configure it at TARGET_WIDTH x TARGET_HEIGHT.
    Falls back gracefully if the requested resolution is not supported.

    Returns an open cv2.VideoCapture object.
    Exits the program with an error message if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(
            "\nERROR: Could not open webcam.\n"
            "Please check that your webcam is connected and not in use by another application.\n"
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  [CAM]  Opened webcam at {actual_w}x{actual_h}")

    return cap


# =============================================================================
# GESTURE CLASSIFIER
# =============================================================================

def _is_finger_extended(lm, tip: int, pip: int, mcp: int) -> bool:
    """
    Determine whether a finger is extended using relative Y-coordinates.

    Strategy:
        A finger is considered 'extended' when its TIP landmark is above its
        MCP (knuckle) landmark in the image frame.

        MediaPipe Y-coordinates increase downward, so 'above' means a smaller
        Y value. We require TIP.y < PIP.y < MCP.y for a clean extension,
        avoiding false positives from partially-curled fingers.

    Args:
        lm   : list of NormalizedLandmark objects
        tip  : landmark index for the fingertip
        pip  : landmark index for the proximal inter-phalangeal joint
        mcp  : landmark index for the metacarpophalangeal joint (knuckle)

    Returns:
        True if the finger is extended, False otherwise.
    """
    tip_y = lm[tip].y
    pip_y = lm[pip].y
    mcp_y = lm[mcp].y

    # Tip must be above PIP, which must be above MCP (in image space)
    return tip_y < pip_y and pip_y < mcp_y


def _is_thumb_extended(lm) -> bool:
    """
    Thumb extension is assessed differently because the thumb moves laterally.

    We compare the lateral (X-axis) distance between THUMB_TIP and THUMB_MCP
    relative to the hand's overall width (wrist to middle-finger-MCP).
    If the tip is sufficiently far to the side, the thumb is extended.

    This works for both left and right hands because we use absolute distance.
    """
    hand_size = abs(lm[MIDDLE_MCP].x - lm[WRIST].x)
    if hand_size < 1e-6:
        return False

    # Lateral distance from thumb tip to thumb CMC base
    lateral = abs(lm[THUMB_TIP].x - lm[THUMB_CMC].x)

    # Also verify the tip is higher (smaller Y) than the IP joint so we
    # don't fire when the thumb is folded flat across the palm
    vertical_ok = lm[THUMB_TIP].y < lm[THUMB_IP].y

    return vertical_ok and (lateral / hand_size > 0.35)


def classify_gesture(hand_landmarks) -> str:
    """
    Classify a hand gesture from MediaPipe hand landmarks.

    Uses relative landmark geometry (no absolute pixel coordinates) so the
    classifier works regardless of hand size or distance from the camera.

    Gesture priority (evaluated top to bottom):
        THUMBS_UP  -> thumb up, all four fingers folded
        POINTING   -> index only extended
        RODENT     -> middle only extended  (displayed as RODENT)
        PEACE      -> index + middle extended, ring + pinky folded
        SOYJAK     -> all four fingers extended (open palm)
        NEUTRAL    -> anything else (fist, ambiguous, no hand)

    Args:
        hand_landmarks : mediapipe NormalizedLandmarkList

    Returns:
        Gesture name string.
    """
    lm = hand_landmarks.landmark

    thumb_up  = _is_thumb_extended(lm)
    index_up  = _is_finger_extended(lm, INDEX_TIP,  INDEX_PIP,  INDEX_MCP)
    middle_up = _is_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_up   = _is_finger_extended(lm, RING_TIP,   RING_PIP,   RING_MCP)
    pinky_up  = _is_finger_extended(lm, PINKY_TIP,  PINKY_PIP,  PINKY_MCP)

    # THUMBS_UP: thumb up, all four fingers folded
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "THUMBS_UP"

    # POINTING: only index finger extended
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINTING"

    # RODENT: only middle finger extended
    if middle_up and not index_up and not ring_up and not pinky_up:
        return "RODENT"

    # PEACE: index + middle extended, ring + pinky folded
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    # SOYJAK (open palm): all four fingers extended
    if index_up and middle_up and ring_up and pinky_up:
        return "SOYJAK"

    # NEUTRAL: closed fist, ambiguous, or any other combination
    return "NEUTRAL"


# =============================================================================
# GESTURE SMOOTHER
# =============================================================================

class GestureSmoother:
    """
    Temporal smoother that commits to a new gesture only after it has been
    detected for STABLE_FRAMES consecutive frames.

    This prevents the meme panel from flickering on noisy detections.
    """

    def __init__(self, stable_frames: int = STABLE_FRAMES):
        self.stable_frames = stable_frames
        self.confirmed     = "NEUTRAL"   # the gesture currently displayed
        self.candidate     = "NEUTRAL"   # the gesture being counted up
        self.count         = 0

    def update(self, raw_gesture: str) -> str:
        """
        Feed in the raw gesture for the current frame.
        Returns the currently confirmed (stable) gesture.
        """
        if raw_gesture == self.candidate:
            self.count += 1
        else:
            self.candidate = raw_gesture
            self.count     = 1

        if self.count >= self.stable_frames:
            self.confirmed = self.candidate

        return self.confirmed


# =============================================================================
# DRAWING UTILITIES
# =============================================================================

_MP_HANDS     = mp.solutions.hands
_MP_FACE_MESH = mp.solutions.face_mesh


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks, frame_w: int, frame_h: int):
    """
    Draw hand skeleton on the frame with a minimal green cyber aesthetic.

    Steps:
        1. Build a cheap glow layer by drawing thick lines, blurring, then
           additively blending onto the frame.
        2. Draw crisp thin connector lines on top.
        3. Draw filled landmark circles with a bright-green ring.
    """
    # Collect pixel coordinates for all 21 landmarks
    pts = []
    for lm in hand_landmarks.landmark:
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        pts.append((px, py))

    # Glow approximation: thick lines -> blur -> additive blend
    glow = np.zeros_like(frame, dtype=np.uint8)
    for start_idx, end_idx in _MP_HANDS.HAND_CONNECTIONS:
        cv2.line(glow, pts[start_idx], pts[end_idx], (0, 160, 50), 5)
    glow = cv2.GaussianBlur(glow, (9, 9), 0)
    cv2.add(frame, glow, dst=frame)

    # Crisp connector lines
    for start_idx, end_idx in _MP_HANDS.HAND_CONNECTIONS:
        cv2.line(frame, pts[start_idx], pts[end_idx], COLOR_GREEN, 1, cv2.LINE_AA)

    # Landmark dots
    for px, py in pts:
        cv2.circle(frame, (px, py), 4, COLOR_GREEN,       -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (0, 255, 120),      1, cv2.LINE_AA)


def draw_face_landmarks(frame: np.ndarray, face_landmarks, frame_w: int, frame_h: int):
    """
    Draw face mesh landmarks with a minimal, technical aesthetic.

    We draw only FACEMESH_CONTOURS (a sparse subset of the 468-point mesh)
    to keep rendering lightweight and visually clean.
    """
    lm = face_landmarks.landmark

    # Tiny dim dot at every landmark
    for landmark in lm:
        px = int(landmark.x * frame_w)
        py = int(landmark.y * frame_h)
        cv2.circle(frame, (px, py), 1, COLOR_GREEN_DIM, -1, cv2.LINE_AA)

    # FACEMESH_CONTOURS for a clean face outline
    for start_idx, end_idx in _MP_FACE_MESH.FACEMESH_CONTOURS:
        p1 = (int(lm[start_idx].x * frame_w), int(lm[start_idx].y * frame_h))
        p2 = (int(lm[end_idx].x   * frame_w), int(lm[end_idx].y   * frame_h))
        cv2.line(frame, p1, p2, COLOR_GREEN_DIM, 1, cv2.LINE_AA)


def draw_ui_overlay(frame: np.ndarray, gesture: str, fps: float):
    """
    Render the top-left HUD text (GESTURE label + FPS counter).

    A semi-transparent black rectangle is drawn first so the text remains
    legible over any webcam background.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (360, 82), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Gesture label
    cv2.putText(
        frame, f"GESTURE: {gesture}",
        (18, 40), FONT, 0.72, COLOR_GREEN, 2, cv2.LINE_AA
    )

    # FPS counter (smaller, dimmer)
    cv2.putText(
        frame, f"FPS: {fps:.0f}",
        (18, 68), FONT, 0.50, COLOR_GREEN_DIM, 1, cv2.LINE_AA
    )

    # Border around webcam panel
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_GREEN_DIM, 1)


# =============================================================================
# MEME PANEL
# =============================================================================

def _fit_image(img: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    """
    Scale img (BGRA) to fit within panel_w x panel_h while preserving aspect
    ratio. Never upscales beyond the image's native resolution.
    """
    ih, iw = img.shape[:2]
    scale = min(panel_w / iw, panel_h / ih, 1.0)  # cap at 1.0 to avoid upscaling
    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _composite_bgra(background: np.ndarray, overlay: np.ndarray, x: int, y: int):
    """
    Alpha-composite a BGRA overlay onto a BGR background at position (x, y).
    """
    oh, ow = overlay.shape[:2]
    bh, bw = background.shape[:2]

    # Clamp ROI
    x1 = max(x, 0);  y1 = max(y, 0)
    x2 = min(x + ow, bw);  y2 = min(y + oh, bh)
    if x2 <= x1 or y2 <= y1:
        return

    ox1 = x1 - x;  oy1 = y1 - y
    ox2 = ox1 + (x2 - x1);  oy2 = oy1 + (y2 - y1)

    roi     = background[y1:y2, x1:x2].astype(np.float32)
    ov_crop = overlay[oy1:oy2, ox1:ox2]

    if overlay.shape[2] == 4:
        alpha   = ov_crop[:, :, 3:4].astype(np.float32) / 255.0
        bgr     = ov_crop[:, :, :3].astype(np.float32)
        blended = bgr * alpha + roi * (1.0 - alpha)
        background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = ov_crop[:, :, :3]


def _add_grain(panel: np.ndarray, intensity: float = 8.0) -> np.ndarray:
    """Add subtle film-grain noise to the panel for a CRT aesthetic."""
    noise = np.random.normal(0, intensity, panel.shape).astype(np.int16)
    return np.clip(panel.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def draw_meme_panel(
    gesture: str,
    assets: dict,
    panel_w: int,
    panel_h: int,
    add_grain: bool = True,
) -> np.ndarray:
    """
    Render the right-side meme panel (pure black + centred Wojak image).

    Returns a BGR numpy array of shape (panel_h, panel_w, 3).
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    img = assets.get(gesture)

    if img is None:
        # Fallback text when the PNG is missing
        text = "IMAGE NOT FOUND"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.7, 1)
        cv2.putText(panel, text, ((panel_w - tw) // 2, panel_h // 2),
                    FONT, 0.7, COLOR_GRAY, 1, cv2.LINE_AA)
        sub = f"[{gesture}.png missing]"
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.42, 1)
        cv2.putText(panel, sub, ((panel_w - sw) // 2, panel_h // 2 + 28),
                    FONT, 0.42, COLOR_GRAY, 1, cv2.LINE_AA)
    else:
        fitted  = _fit_image(img, panel_w - 20, panel_h - 60)
        fh, fw  = fitted.shape[:2]
        ox = (panel_w - fw) // 2
        oy = (panel_h - fh) // 2
        _composite_bgra(panel, fitted, ox, oy)

    # Gesture label at bottom of panel
    label = f"[ {gesture} ]"
    (lw, _), _ = cv2.getTextSize(label, FONT, 0.6, 1)
    cv2.putText(
        panel, label,
        ((panel_w - lw) // 2, panel_h - 14),
        FONT, 0.6, COLOR_GREEN, 1, cv2.LINE_AA,
    )

    # Divider line on left edge
    cv2.line(panel, (0, 0), (0, panel_h - 1), COLOR_GREEN_DIM, 1)

    if add_grain:
        panel = _add_grain(panel, intensity=8)

    return panel


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    print("\n+======================================+")
    print("|   WOJAK GESTURE CONTROLLER  v1.0    |")
    print("+======================================+\n")

    print("[*] Loading assets ...")
    assets = load_assets()
    print()

    print("[*] Initialising camera ...")
    cap = initialize_camera()
    print()

    print("[*] Initialising MediaPipe ...")
    try:
        mp_hands_mod = mp.solutions.hands
        hands = mp_hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        )

        mp_face_mod = mp.solutions.face_mesh
        face_mesh = mp_face_mod.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE,
        )
    except Exception as exc:
        print(f"\nERROR: Could not initialise MediaPipe: {exc}")
        cap.release()
        sys.exit(1)

    print("[*] Ready. Press ESC to exit.\n")

    smoother  = GestureSmoother(stable_frames=STABLE_FRAMES)
    prev_time = time.perf_counter()

    WINDOW_NAME = "Wojak Gesture Controller"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ERROR: Failed to read frame from webcam. Exiting.")
                break

            # Mirror so it feels like looking in a mirror
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Split screen widths
            meme_w = int(frame_w * MEME_PANEL_RATIO)
            cam_w  = frame_w - meme_w
            cam_frame = frame[:, :cam_w].copy()

            # Convert to RGB once for MediaPipe
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            rgb.flags.writeable = True

            # Classify gesture
            raw_gesture = "NEUTRAL"
            if hand_results.multi_hand_landmarks:
                raw_gesture = classify_gesture(hand_results.multi_hand_landmarks[0])

            confirmed_gesture = smoother.update(raw_gesture)

            # Draw face mesh first (underneath hand landmarks)
            if face_results.multi_face_landmarks:
                draw_face_landmarks(
                    cam_frame,
                    face_results.multi_face_landmarks[0],
                    cam_w,
                    frame_h,
                )

            # Draw hand skeleton
            if hand_results.multi_hand_landmarks:
                draw_hand_landmarks(
                    cam_frame,
                    hand_results.multi_hand_landmarks[0],
                    cam_w,
                    frame_h,
                )

            # FPS
            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_ui_overlay(cam_frame, confirmed_gesture, fps)

            meme_panel = draw_meme_panel(
                confirmed_gesture, assets, meme_w, frame_h
            )

            combined = np.hstack([cam_frame, meme_panel])
            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                print("\n[*] ESC pressed - exiting ...")
                break

    finally:
        cap.release()
        hands.close()
        face_mesh.close()
        cv2.destroyAllWindows()
        print("[*] Resources released. Goodbye.\n")


if __name__ == "__main__":
    main()
