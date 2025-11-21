import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Initialize TFLite model
def load_tflite_model(model_path):
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Model loaded successfully")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        exit(1)

# Initialize MediaPipe hands
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return hands, mp_hands, mp_drawing, mp_drawing_styles

def process_landmarks(landmarks, mp_hands_obj, handedness="Right"):
    # Initialize feature array
    features = []
    
    # 1. Absolute normalized coordinates (21 landmarks x 3 coordinates = 63 features)
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    # Keep a copy of original coordinates for bounding box
    x_ = [lm.x for lm in landmarks]
    y_ = [lm.y for lm in landmarks]
    
    # 2. Normalize coordinates to wrist (already stored absolute values above)
    wrist = landmarks[mp_hands_obj.HandLandmark.WRIST]
    
    # 3. Tip distances (5 features)
    for tip_id in [mp_hands_obj.HandLandmark.THUMB_TIP, 
                  mp_hands_obj.HandLandmark.INDEX_FINGER_TIP,
                  mp_hands_obj.HandLandmark.MIDDLE_FINGER_TIP,
                  mp_hands_obj.HandLandmark.RING_FINGER_TIP,
                  mp_hands_obj.HandLandmark.PINKY_TIP]:
        tip = landmarks[tip_id]
        # Calculate distance from wrist to tip
        distance = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
        features.append(distance)
    
    # 4. Angles between joints (5 features)
    def calculate_angle(p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm > 0 and v2_norm > 0:
            dot_product = np.dot(v1, v2) / (v1_norm * v2_norm)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            return np.arccos(dot_product)
        return 0

    for finger_base, finger_mid, finger_tip in [
        (mp_hands_obj.HandLandmark.THUMB_CMC, mp_hands_obj.HandLandmark.THUMB_MCP, mp_hands_obj.HandLandmark.THUMB_TIP),
        (mp_hands_obj.HandLandmark.INDEX_FINGER_MCP, mp_hands_obj.HandLandmark.INDEX_FINGER_PIP, mp_hands_obj.HandLandmark.INDEX_FINGER_TIP),
        (mp_hands_obj.HandLandmark.MIDDLE_FINGER_MCP, mp_hands_obj.HandLandmark.MIDDLE_FINGER_PIP, mp_hands_obj.HandLandmark.MIDDLE_FINGER_TIP),
        (mp_hands_obj.HandLandmark.RING_FINGER_MCP, mp_hands_obj.HandLandmark.RING_FINGER_PIP, mp_hands_obj.HandLandmark.RING_FINGER_TIP),
        (mp_hands_obj.HandLandmark.PINKY_MCP, mp_hands_obj.HandLandmark.PINKY_PIP, mp_hands_obj.HandLandmark.PINKY_TIP)
    ]:
        base = landmarks[finger_base]
        mid = landmarks[finger_mid]
        tip = landmarks[finger_tip]
        angle = calculate_angle(base, mid, tip)
        features.append(angle)
    
    # 5. Handedness (1 feature)
    # Detect if hand is left or right based on thumb position relative to pinky
    detected_handedness = "Right" if landmarks[mp_hands_obj.HandLandmark.THUMB_TIP].x > landmarks[mp_hands_obj.HandLandmark.PINKY_TIP].x else "Left"
    handedness_value = 1.0 if detected_handedness == "Right" else 0.0
    features.append(handedness_value)
    
    # Verify feature count
    assert len(features) == 74, f"Expected 74 features, got {len(features)}"
    
    return np.array([features], dtype=np.float32), x_, y_
def main():
    # Define gesture classes (update based on your model)
    gesture_classes = ["1", "2", "3", "Bye", "Hello", "Thank You"]
    
    # Load TFLite model
    model_path = "sign_language_model.tflite"  # Update this path
    interpreter, input_details, output_details = load_tflite_model(model_path)
    
    # Check if model expects 74 features
    expected_feature_count = input_details[0]['shape'][1]
    if expected_feature_count != 74:
        print(f"⚠️ WARNING: Model expects {expected_feature_count} features, but code is configured for 74 features!")
        print("Please make sure your model matches your feature extraction.")
    else:
        print(f"✅ Model input shape verified: Expecting {expected_feature_count} features")
    
    # Initialize MediaPipe
    hands, mp_hands, mp_drawing, mp_drawing_styles = initialize_mediapipe()
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Trying alternative...")
        # Try alternative camera index
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open any webcam")
            exit()
    
    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera resolution set to: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # For FPS calculation
    prev_time = 0
    
    # Prediction stabilization variables
    prediction = None
    confidence = None
    prediction_threshold = 5
    last_predictions = []
    
    print("Press 'q' to exit")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error reading from webcam")
            break
        
        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Flip the image horizontally for selfie-view
        image = cv2.flip(image, 1)
        
        # Get image dimensions
        H, W, _ = image.shape
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_image)
        
        # Display FPS
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "Press Q to Exit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process hands if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get handedness from results if available
                handedness = "Right"
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                
                # Process landmarks for TFLite input
                processed_data, x_, y_ = process_landmarks(hand_landmarks.landmark, mp_hands, handedness)
                
                # Calculate bounding box
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                try:
                    # Verify input shape
                    feature_count = processed_data.shape[1]
                    cv2.putText(image, f"Features: {feature_count}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Set input tensor
                    interpreter.set_tensor(input_details[0]['index'], processed_data)
                    
                    # Run inference
                    interpreter.invoke()
                    
                    # Get output tensor
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Get prediction
                    probs = output_data[0]
                    pred_idx = np.argmax(probs)
                    pred_class = gesture_classes[pred_idx]
                    pred_confidence = probs[pred_idx]
                    
                    # Stabilize prediction
                    last_predictions.append(pred_class)
                    if len(last_predictions) > prediction_threshold:
                        last_predictions.pop(0)
                    
                    # Only update prediction if consistent
                    if len(last_predictions) == prediction_threshold and all(p == last_predictions[0] for p in last_predictions):
                        prediction = pred_class
                        confidence = pred_confidence
                    
                    # Display prediction
                    if prediction is not None:
                        label = f"{prediction} ({confidence:.2f})"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                        
                        # Ensure text doesn't go off screen
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
                        
                        cv2.putText(image, label, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                        
                except Exception as e:
                    print(f"Inference error: {e}")
                    cv2.putText(image, f"Error: {str(e)[:20]}...", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Reset predictions when no hand is detected
            last_predictions = []
            cv2.putText(image, "No hand detected", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow('Hand Gesture Recognition', image)
        
        # Quit on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()