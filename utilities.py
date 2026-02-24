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
        print("❌ You've used all 3 guesses!")
        return [], None, "No more guesses", 0

    _scale_guesses_used += 1

    # Load & scale
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    scaled = im.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    temp = _make_temp_image(scaled)

    # Run detection
    dets, ann, _ = detect_wally(temp)

    if len(dets) == 0:
        # AI broken - score based on how close to optimal breaking point (0.32)
        if 0.30 <= scale <= 0.33:
            msg = f"🎯 PERFECT! AI broken at {scale} - Optimal breaking point!"
            points = 10
        elif 0.28 <= scale <= 0.35:
            msg = f"⭐ EXCELLENT! AI broken at {scale} - Very close to optimal!"
            points = 8
        elif 0.25 <= scale <= 0.40:
            msg = f"👍 GOOD! AI broken at {scale} - Effective!"
            points = 6
        else:
            msg = f"⚠️ AI broken at {scale} - But too small (overkill)!"
            points = 4
        return dets, ann, msg, points
    else:
        # AI survived
        conf = dets[0][1]
        msg = f"Guess {_scale_guesses_used}: AI survived at {scale} (conf={conf:.2f}) - Try smaller!"
        points = 2
        return dets, ann, msg, points

def calculate_scale_final_score():
    """Call this after all 3 guesses to get the final score"""
    # This gets called from the notebook to sum up all attempts
    pass

def scale_challenge_complete(attempt_scores):
    """
    Calculate final score for scale challenge.
    attempt_scores: list of points from each attempt
    """
    total_score = max(attempt_scores)  # Take the best single attempt
    
    print("\n🏆 SCALE CHALLENGE RESULTS:")
    print("-" * 40)
    
    for i, score in enumerate(attempt_scores, 1):
        if score == 10:
            print(f"Attempt {i}: 🎯 PERFECT PRECISION! ({score} pts)")
        elif score == 8:
            print(f"Attempt {i}: ⭐ CLOSE TO OPTIMAL! ({score} pts)")
        elif score == 6:
            print(f"Attempt {i}: ✅ GOOD BREAK! ({score} pts)")
        elif score == 4:
            print(f"Attempt {i}: ⚠️ OVERKILL BREAK! ({score} pts)")
        elif score == 2:
            print(f"Attempt {i}: 💪 AI SURVIVED ({score} pts)")
        else:
            print(f"Attempt {i}: Invalid attempt ({score} pts)")
    
    print(f"\n🏆 Best Score: {total_score}/10 points")
    
    if total_score == 10:
        print("🎉 PERFECT! You found the optimal breaking point!")
    elif total_score == 8:
        print("🌟 EXCELLENT! Very close to the optimal scale!")
    elif total_score == 6:
        print("👍 GOOD! You broke the AI effectively!")
    elif total_score == 4:
        print("⚠️ OVERKILL! You broke it but went too small - be more precise!")
    else:
        print("💪 The AI survived all your attempts!")
    
    return total_score


# -----------------------------------------------------------
# OCCLUSION / BLUR ATTACK
# -----------------------------------------------------------

def blur_attack(img_path, blur_strength):
    """
    Apply blur attack to Waldo's detection area only.
    Returns detection results and scoring based on optimal blur threshold.
    """
    MINIMUM_BLUR_THRESHOLD = 25  # Updated based on test results
    
    # Ensure blur strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
        print(f"Adjusted to odd number: {blur_strength}")
    
    # First, find Waldo in the original image
    original_dets, _, _ = detect_wally(img_path)
    if len(original_dets) == 0:
        return [], None, "Could not find Wally in original image to blur", 0
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        return [], None, "Error: Could not load image", 0
    
    # Get Waldo's bounding box and blur only that region
    waldo_box = original_dets[0][0]  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, waldo_box)
    
    # Create copy and blur only Waldo region
    blurred_img = img.copy()
    waldo_region = blurred_img[y1:y2, x1:x2]
    
    # Check if region is valid
    if waldo_region.size == 0:
        return [], None, "Error: Invalid Wally region", 0
    
    blurred_waldo = cv2.GaussianBlur(waldo_region, (blur_strength, blur_strength), 0)
    blurred_img[y1:y2, x1:x2] = blurred_waldo
    
    temp_path = _make_temp_image_from_array(blurred_img)
    
    # Run detection on image with blurred Waldo
    dets, ann, _ = detect_wally(temp_path)
    os.unlink(temp_path)  # Clean up
    
    # Calculate score and message
    points = 0
    if len(dets) == 0:
        # AI failed to detect blurred Waldo - SUCCESS! Score based on precision to blur 25
        if blur_strength == MINIMUM_BLUR_THRESHOLD:
            msg = f"PERFECT! Optimal Wally blur at {blur_strength} - AI broken with precision!"
            points = 10
        elif 23 <= blur_strength <= 27:  # Within 2 of optimal (25)
            msg = f"Excellent! AI broken with blur {blur_strength} - Very close to optimal "
            points = 8
        elif 20 <= blur_strength <= 30:  # Within 5 of optimal (25)
            msg = f"Good! AI broken with blur {blur_strength} - Close to optimal "
            points = 6
        else:
            msg = f"Success! AI broken with blur {blur_strength} but could be more precise"
            points = 4
    else:
        # AI still detected blurred Waldo - FAILURE
        conf = dets[0][1]
        msg = f"AI survived Wally blur {blur_strength} (confidence: {conf:.2f}) - Try stronger blur!"
        points = 2
    
    return dets, ann, msg, points

def _make_temp_image_from_array(img_array):
    """Helper to save opencv array as temporary image"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, img_array)
        return tmp.name

# Calculate final score
def blur_challenge_complete(blur_scores, blur_attempts):
    print("\n🏆 BLUR ATTACK RESULTS:")
    print("-" * 40)
    
    for i, attempt in enumerate(blur_attempts, 1):
        if attempt['points'] == 10:
            print(f"Attempt {i}: 🎯 PERFECT PRECISION! Blur {attempt['blur']} ({attempt['points']} pts)")
        elif attempt['points'] == 8:
            print(f"Attempt {i}: ⭐ VERY CLOSE! Blur {attempt['blur']} ({attempt['points']} pts)")
        elif attempt['points'] == 6:
            print(f"Attempt {i}: ✅ CLOSE! Blur {attempt['blur']} ({attempt['points']} pts)")
        elif attempt['points'] == 4:
            print(f"Attempt {i}: ⚠️ BREAK BUT NOT OPTIMAL! Blur {attempt['blur']} ({attempt['points']} pts)")
        elif attempt['points'] == 2:
            print(f"Attempt {i}: 💪 AI SURVIVED Blur {attempt['blur']} ({attempt['points']} pts)")
    
    final_score = max(blur_scores)
    print(f"\n🏆 Best Score: {final_score}/10 points")
    
    successful_breaks = [a for a in blur_attempts if not a['detected']]
    if successful_breaks:
        min_successful = min(a['blur'] for a in successful_breaks)
        print(f"✅ Minimum successful blur: {min_successful} (target: 25)")  # Changed from 19 to 25
    
    # Add final messages based on best score
    if final_score == 10:
        print("🎉 PERFECT! You found the exact minimal blur!")
    elif final_score >= 8:
        print("🌟 EXCELLENT! Very close to optimal!")
    elif final_score >= 6:
        print("👍 GOOD! Close to the target!")
    elif final_score >= 4:
        print("⚠️ You broke it but could be more precise!")
    else:
        print("💪 The AI was too resilient!")
    
    return final_score


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

def waldo_sizing_challenge(base_x, base_y, waldo_image, bg_image):
    """Interactive challenge to find the smallest Waldo the AI can detect"""
    
    print("🎯 Waldo Sizing Challenge!")
    print("Find the smallest Waldo HEIGHT the AI can still detect with >50% confidence")
    print("You get 3 attempts. Valid range: 10-300 pixels")
    print()
    
    results = []
    
    for attempt in range(1, 4):
        print(f"--- Attempt {attempt} ---")
        
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
        bg = bg_image.copy()
        waldo_resized = waldo_image.resize((width, height), Image.LANCZOS)
        bg.paste(waldo_resized, (base_x, base_y), waldo_resized)
        
        # Test with YOLO
        dets, annotated, msg = detect_wally(bg)
        confidence = dets[0][1] if len(dets) > 0 else 0
        
        results.append({
            'height': height,
            'width': width, 
            'confidence': confidence,
            'detected': confidence > 0.5  # Lowered threshold based on your data
        })
        
        show(annotated, f"Attempt {attempt}: Waldo {width}x{height}px")
        
        # Better status logic that considers both too small AND too large
        if confidence > 0.5:
            status = "✅ DETECTED"
        elif height < 40:  # Based on your test data - minimum viable size
            status = "❌ TOO SMALL"
        elif height > 200:  # Set a reasonable upper limit
            status = "⚠️ TOO LARGE"
        else:
            status = "❌ NOT DETECTED"
        
        print(f"Size: {width}x{height}px | Confidence: {confidence:.3f} | {status}")
        print()
    
    # ENHANCED SCORING based on your optimal data
    print("🏆 FINAL SCORING:")
    print("-" * 50)
    
    valid_results = [r for r in results if r['detected']]
    
    if not valid_results:
        print("❌ No successful detections - 0 points")
        score = 0
    else:
        smallest_height = min(r['height'] for r in valid_results)
        smallest_confidence = min(r['confidence'] for r in valid_results if r['detected'])
        
        print(f"✅ Smallest successful guess: {smallest_height}px (confidence: {smallest_confidence:.3f})")
        
        # UPDATED SCORING based on your test results:
        # 30px = impossible, 40px = possible, 50px+ = good confidence
        if smallest_height <= 35:
            score = 15  # Bonus for near-impossible detection
            print("🚀 LEGENDARY! Sub-40px detection - 15 points!")
        elif smallest_height <= 40:  # At the limit (24x40px from test)
            score = 12
            print("🌟 INCREDIBLE! Minimum viable size - 12 points!")
        elif smallest_height <= 50:  # 30x50px (first >90% confidence)
            score = 10
            print("🎉 EXCELLENT! High-confidence small Waldo - 10 points!")
        elif smallest_height <= 70:  # Good range
            score = 8
            print("👏 GREAT! Small Waldo detected - 8 points!")
        elif smallest_height <= 100:  # Medium range  
            score = 6
            print("👍 GOOD! Medium sized Waldo detected - 6 points!")
        elif smallest_height <= 130:  # Large but reasonable
            score = 4
            print("✅ OKAY! Large Waldo detected - 4 points!")
        else:  # Very large
            score = 2
            print("📏 Only very large Waldo detected - 2 points!")
        
        # BONUS for high confidence
        max_confidence = max(r['confidence'] for r in valid_results)
        if max_confidence > 0.95:
            score += 2
            print(f"🎯 CONFIDENCE BONUS: +2 points for {max_confidence:.3f} confidence!")
        elif max_confidence > 0.90:
            score += 1
            print(f"🎯 CONFIDENCE BONUS: +1 point for {max_confidence:.3f} confidence!")
    
    print(f"\nFinal Score: {score}/15 points")
    
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