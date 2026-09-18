# Wojak Gesture Controller

A real-time computer-vision application that detects hand gestures via webcam
and dynamically switches Wojak meme images on screen. Built with Python,
OpenCV, and MediaPipe. Inspired by the creative-coding aesthetic of projects
like **mundane.zip**.

```
+-------------------------------+---------------------------+
|                               |                           |
|   WEBCAM  (mirrored)          |   WOJAK MEME              |
|                               |                           |
|   face mesh + hand skeleton   |       [ IMAGE ]           |
|                               |                           |
|  GESTURE: POINTING            |   [ POINTING ]            |
|  FPS: 29                      |                           |
+-------------------------------+---------------------------+
```

---

## Project Structure

```
wojak-gesture-controller/
|
+-- main.py            <- Application entry point
+-- requirements.txt   <- Python dependencies
+-- README.md          <- This file
|
+-- assets/
    +-- neutral.png    <- shown for NEUTRAL / fist / no hand
    +-- pointing.png   <- shown for POINTING gesture
    +-- sojak.png      <- shown for SOYJAK (open palm)
    +-- middle.png     <- shown for RODENT gesture
    +-- thumbs_up.png  <- shown for THUMBS_UP
    +-- peace.png      <- shown for PEACE (two fingers)
```

---

## Installation

### 1. Create a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python main.py
```

---

## Where to Put Your Meme PNGs

Place your PNG files inside the `assets/` folder **next to `main.py`**.

| Filename        | Triggered by gesture |
|-----------------|----------------------|
| `neutral.png`   | NEUTRAL (fist / no hand) |
| `pointing.png`  | POINTING (index finger up) |
| `sojak.png`     | SOYJAK (open palm, all fingers up) |
| `middle.png`    | RODENT (middle finger only) |
| `thumbs_up.png` | THUMBS_UP (thumb up, fist) |
| `peace.png`     | PEACE (index + middle up) |

> PNG files can have transparency (RGBA). The app composites them over the
> black background preserving the alpha channel.
>
> If a file is missing the app will display **IMAGE NOT FOUND** on the right
> panel rather than crashing. You can add images one at a time.

---

## Supported Gestures

| Gesture    | How to perform it                             | Display label |
|------------|-----------------------------------------------|---------------|
| NEUTRAL    | Closed fist, or no hand in frame              | GESTURE: NEUTRAL |
| POINTING   | Index finger extended, others folded          | GESTURE: POINTING |
| SOYJAK     | All four fingers extended (open palm)         | GESTURE: SOYJAK |
| RODENT     | Middle finger only extended                   | GESTURE: RODENT |
| THUMBS_UP  | Thumb extended sideways, all fingers folded   | GESTURE: THUMBS_UP |
| PEACE      | Index + middle extended, ring + pinky folded  | GESTURE: PEACE |

### How finger extension is detected

Each finger is classified as *extended* when its **TIP** landmark is higher
in the frame (smaller Y coordinate) than its **PIP** (middle joint), which is
in turn higher than the **MCP** (knuckle). Because these are relative
comparisons using MediaPipe's normalised `[0.0, 1.0]` coordinates, the
classifier works at any hand size and distance from the camera.

The **thumb** uses a different method because it moves laterally: the
horizontal distance between `THUMB_TIP` and `THUMB_CMC` is compared against
the overall hand width. If the tip is far enough to the side *and* above the
IP joint, the thumb is considered extended.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit the application |

---

## Gesture Stability

The displayed gesture only updates after the **same gesture has been detected
for 5 consecutive frames**. This prevents flickering. You can change the
threshold at the top of `main.py`:

```python
STABLE_FRAMES = 5   # increase for more stability, decrease for faster switching
```

---

## Troubleshooting

### Webcam cannot be opened
```
ERROR: Could not open webcam.
```
- Make sure no other application (Teams, Zoom, browser) is using the camera.
- Try changing `CAMERA_INDEX = 0` to `1` or `2` in `main.py` if you have
  multiple cameras.

### Low FPS
- Close other GPU-intensive applications.
- Face mesh can be disabled by commenting out the `draw_face_landmarks` call
  in `main()` if performance is a concern.
- Run on a machine with a dedicated GPU if available.

### Gestures not detected reliably
- Ensure good, even lighting on your hand.
- Keep your hand within 30-70 cm of the camera.
- Avoid backgrounds that are similar in colour to skin tone.

### mediapipe import errors
- Make sure you activated your virtual environment before running.
- Reinstall: `pip install --force-reinstall mediapipe`
- MediaPipe requires Python 3.8–3.11. Python 3.12 may have compatibility issues.

---

## Requirements

- Python 3.10+
- OpenCV `>= 4.8`
- MediaPipe `>= 0.10`
- NumPy `>= 1.24`
- A connected webcam

---

## License

Bring your own Wojaks. No meme images are bundled with this project.
