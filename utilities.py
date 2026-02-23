import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import tempfile
import uuid

# -----------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------

MODEL_PATH = "runs/detect/train/weights/best.pt"
trained = YOLO(MODEL_PATH)

# scale game state
_SCALE_GUESS_LIMIT = 3
_scale_guesses_used = 0

# temp directory for student files
_temp_dir = tempfile.gettempdir()


# -----------------------------------------------------------
# BASIC UTILITIES
# -----------------------------------------------------------

def _make_temp_image(img_pil):
    """Save a PIL image to a unique temp file and return its path."""
    filename = os.path.join(_temp_dir, f"wally_tmp_{uuid.uuid4()}.jpg")
    img_pil.save(filename)
    return filename


def show(img_bgr, title=""):
    """Display an OpenCV BGR image as RGB in matplotlib."""
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()


# -----------------------------------------------------------
# DETECTION
# -----------------------------------------------------------

def detect_wally(img_path, conf_thr=0.25):
    """Run YOLO detection on an image path."""
    results = trained.predict(img_path, conf=conf_thr, verbose=False)
    result = results[0]
    img = result.orig_img.copy()

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            score = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            detections.append((xyxy, score, cls))

            x1,y1,x2,y2 = map(int, xyxy)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
            cv2.putText(img,f"{score:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

    return detections, img, result


# -----------------------------------------------------------
# SCALE BREAKING GAME (3 guesses only)
# -----------------------------------------------------------

def reset_scale_game():
    """Instructor can reset student attempts."""
    global _scale_guesses_used
    _scale_guesses_used = 0


def scale_guess(img_path, scale):
    """
    Student-only API.
    Students get ONLY 3 attempts.
    They pass a scale (e.g., 0.5), and we test detection.
    """
    global _scale_guesses_used, _SCALE_GUESS_LIMIT

    if _scale_guesses_used >= _SCALE_GUESS_LIMIT:
        return None, None, "❌ No guesses left! (You only get 3)."

    _scale_guesses_used += 1

    # Load & scale
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    scaled = im.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    temp = _make_temp_image(scaled)

    # Run detection
    dets, ann, _ = detect_wally(temp)

    if len(dets) == 0:
        msg = f"💥 Guess {_scale_guesses_used}: AI FAILED at scale {scale}! You broke it!"
        return dets, ann, msg
    else:
        conf = dets[0][1]
        msg = f"🤖 Guess {_scale_guesses_used}: AI survived at {scale} (conf={conf:.2f}). Try smaller!"
        return dets, ann, msg


# -----------------------------------------------------------
# OCCLUSION / BLUR ATTACK
# -----------------------------------------------------------

def blur_attack(img_path, size=60):
    """
    Automatically detects Wally, then applies a blur on top of him.
    """
    dets, ann, _ = detect_wally(img_path)
    if len(dets) == 0:
        return None, None, "⚠️ Wally not detected in the base image."

    (x1,y1,x2,y2), score, _ = dets[0]
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)

    # Load image for blurring
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    bx1, by1 = max(0, cx-size), max(0, cy-size)
    bx2, by2 = min(W, cx+size), min(H, cy+size)

    attacked = img.copy()
    attacked[by1:by2, bx1:bx2] = cv2.GaussianBlur(attacked[by1:by2, bx1:bx2], (51,51), 0)

    temp = os.path.join(_temp_dir, f"wally_blur_{uuid.uuid4()}.jpg")
    cv2.imwrite(temp, attacked)

    dets2, ann2, _ = detect_wally(temp)
    return dets2, ann2, f"Blur attack applied. Before: {score:.2f}, After: {[d[1] for d in dets2] if dets2 else 0}"


# -----------------------------------------------------------
# AUGMENTATIONS
# -----------------------------------------------------------

def augmented_versions(img_path):
    """
    Returns paths to original + augmented versions:
    bright, dark, contrast, flipped.
    """
    img = Image.open(img_path).convert("RGB")

    variants = {
        "original": img,
        "bright": ImageEnhance.Brightness(img).enhance(1.5),
        "dark": ImageEnhance.Brightness(img).enhance(0.6),
        "contrast": ImageEnhance.Contrast(img).enhance(1.4),
        "flip": ImageOps.mirror(img),
    }

    out = {}
    for name, im in variants.items():
        p = os.path.join(_temp_dir, f"wally_aug_{name}_{uuid.uuid4()}.jpg")
        im.save(p)
        out[name] = p

    return out


# -----------------------------------------------------------
# LEADERBOARD
# -----------------------------------------------------------

_LEADERBOARD = {}

def add_points(team, pts):
    _LEADERBOARD[team] = _LEADERBOARD.get(team, 0) + pts

def show_leaderboard():
    print("\n🏆 Leaderboard")
    for team, score in sorted(_LEADERBOARD.items(), key=lambda x:-x[1]):
        print(f"{team}: {score} pts")