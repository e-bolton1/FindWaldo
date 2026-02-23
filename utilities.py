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
    results = trained.predict(img_path, conf=conf_thr, verbose=False)
    result = results[0]
    img = result.orig_img.copy()

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        # Get only the box with highest confidence
        confidences = result.boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()
        box = result.boxes[best_idx]
        
        xyxy = box.xyxy.cpu().numpy()[0]
        conf = float(box.conf.cpu().numpy()[0])
        cls = int(box.cls.cpu().numpy()[0])
        detections.append((xyxy, conf, cls))

        x1,y1,x2,y2 = map(int, xyxy)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

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
        msg = f"Guess {_scale_guesses_used}: AI couldn't find Wally at scale {scale}! You broke it!"
        return dets, ann, msg
    else:
        conf = dets[0][1]
        msg = f"Guess {_scale_guesses_used}: AI detected something at {scale} (conf={conf:.2f})"
        return dets, ann, msg


# -----------------------------------------------------------
# OCCLUSION / BLUR ATTACK
# -----------------------------------------------------------

def blur_attack(img_path, blur_strength=51):
    """
    Automatically detects Wally, then applies a blur on top of him.
    
    Args:
        blur_strength: Intensity of blur (must be odd number, higher = more blur)
    """
    dets, ann, _ = detect_wally(img_path)
    if len(dets) == 0:
        return None, None, "⚠️ Wally not detected in the base image."

    (x1,y1,x2,y2), score, _ = dets[0]
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)

    # Load image for blurring
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    size = 60  # fixed region size
    bx1, by1 = max(0, cx-size), max(0, cy-size)
    bx2, by2 = min(W, cx+size), min(H, cy+size)

    attacked = img.copy()
    # Ensure blur_strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    attacked[by1:by2, bx1:bx2] = cv2.GaussianBlur(attacked[by1:by2, bx1:bx2], (blur_strength, blur_strength), 0)

    temp = os.path.join(_temp_dir, f"wally_blur_{uuid.uuid4()}.jpg")
    cv2.imwrite(temp, attacked)

    dets2, ann2, _ = detect_wally(temp)
    after_conf = dets2[0][1] if len(dets2) > 0 else 0
    return dets2, ann2, f"Blur attack (strength={blur_strength}). Before: {score:.2f}, After: {after_conf:.2f}"


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


# Add these two functions to utilities.py

def display_images_for_ranking(image_list):
    """
    Display images and get user's predicted confidence scores.
    Returns the list of predicted confidences.
    """
    import cv2
    import numpy as np
    
    # Step 1: Display all images WITHOUT confidence scores
    print("🖼️ Here are the images. Study them carefully!\n")
    for i, img_path in enumerate(image_list, 1):
        img = Image.open(img_path)
        show(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), f"Image {i}: {img_path}")

    # Step 2: Ask user to predict confidence scores
    print("\n" + "="*60)
    print("Predict the AI's confidence score for EACH image!")
    print("="*60)
    print("\n Enter confidence as a percentage (0-100)")
    print("Example: If you think AI is 85% confident, enter: 85\n")

    predicted_confidences = []
    for i, img_path in enumerate(image_list, 1):
        conf_input = float(input(f"Predicted confidence for Image {i} ({img_path}): "))
        predicted_confidences.append(conf_input)
    
    return predicted_confidences


def evaluate_ranking_predictions(image_list, predicted_confidences):
    """
    Run AI detection, compare with predictions, and award points.
    Returns total points earned.
    """
    import cv2
    
    # Step 1: Get actual AI confidence scores
    print("\n🤖 Running AI detection...\n")
    scores = []
    for img_path in image_list:
        dets, img, result = detect_wally(img_path)
        show(img, f"AI Detection: {img_path}")
        
        if len(dets) > 0:
            scores.append((img_path, dets[0][1] * 100))  # Convert to percentage
        else:
            scores.append((img_path, 0))

    # Step 2: Display results and calculate points
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)

    total_points = 0
    for i, (img_path, actual_conf) in enumerate(scores):
        predicted_conf = predicted_confidences[i]
        difference = abs(predicted_conf - actual_conf)
        
        print(f"\nImage {i+1}: {img_path}")
        print(f"  👤 Your prediction: {predicted_conf:.1f}%")
        print(f"  🤖 Actual AI confidence: {actual_conf:.1f}%")
        print(f"  📏 Difference: {difference:.1f}%")
        
        # Award points based on accuracy
        if difference <= 5:
            points = 10
            print(f"  🎯 EXCELLENT! Within 5%! +10 points!")
        elif difference <= 10:
            points = 7
            print(f"  ✅ Good! Within 10%! +7 points!")
        elif difference <= 15:
            points = 5
            print(f"  👍 Fair! Within 15%! +5 points!")
        elif difference <= 20:
            points = 3
            print(f"  👌 Close! Within 20%! +3 points!")
        else:
            points = 0
            print(f"  ❌ Too far off. +0 points")
        
        total_points += points

    print(f"\n{'='*60}")
    print(f"🏆 TOTAL SCORE: {total_points}/{len(image_list) * 10} points")
    print(f"{'='*60}")
    
    return total_points


# -----------------------------------------------------------
# Make different sizes of Waldo for scale-breaking game
# -----------------------------------------------------------

def waldo_sizing_challenge(base_x, base_y, waldo_sprite, bg_full):
    """Interactive challenge to find the smallest Waldo the AI can detect"""
    
    print("🎯 Waldo Sizing Challenge!")
    print("Find the smallest Waldo HEIGHT the AI can still detect with >90% confidence")
    print("You get 2 attempts. Valid range: 10-300 pixels")
    print("(Width automatically calculated to maintain proportions)")
    print()
    
    results = []
    
    for attempt in range(1, 3):
        print(f"--- Attempt {attempt} ---")
        
        # Get valid height input
        while True:
            try:
                height = int(input(f"Enter Waldo HEIGHT (10-300): "))
                if 10 <= height <= 300:
                    break
                print(f"❌ Must be between 10-300 pixels!")
            except ValueError:
                print("❌ Please enter a valid number!")
        
        # Calculate proportional width and create image
        width = int(height * 0.6)
        bg = bg_full.copy()
        waldo_resized = waldo_sprite.resize((width, height), Image.LANCZOS)
        bg.paste(waldo_resized, (base_x, base_y), waldo_resized)
        
        # Test with AI
        dets, annotated, msg = detect_wally(bg)
        confidence = dets[0][1] if len(dets) > 0 else 0
        
        results.append({
            'height': height,
            'width': width, 
            'confidence': confidence,
            'detected': confidence > 0.9
        })
        
        # Show result
        show(annotated, f"Attempt {attempt}: {width}x{height} - Confidence: {confidence:.3f}")
        status = "✅ DETECTED" if confidence > 0.9 else "❌ TOO SMALL"
        print(f"Size: {width}x{height} | Confidence: {confidence:.3f} | {status}")
        print()
    
    # Calculate score
    print("🏆 FINAL SCORING:")
    print("-" * 30)
    
    valid_results = [r for r in results if r['detected']]
    
    if not valid_results:
        print("❌ No successful detections - 0 points")
        score = 0
    else:
        smallest_height = min(r['height'] for r in valid_results)
        print(f"✅ Smallest successful height: {smallest_height}px")
        
        if smallest_height <= 40:
            score = 10
            print("🌟 AMAZING! Tiny Waldo - 10 points!")
        elif smallest_height <= 60:
            score = 8  
            print("🎉 EXCELLENT! Small Waldo - 8 points!")
        elif smallest_height <= 80:
            score = 6
            print("👍 GOOD! Medium Waldo - 6 points!")
        elif smallest_height <= 100:
            score = 4
            print("✅ DECENT! Large Waldo - 4 points!")
        else:
            score = 2
            print("📏 Very large Waldo only - 2 points!")
    
    print(f"\nFinal Score: {score}/10 points")
    return score, results




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