import os
import pickle
import numpy as np
import mediapipe as mp
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    min_detection_confidence=0.2,  
    max_num_hands=1
)

DATA_DIR = './data'
SKIPPED_DIR = './skipped_images'
os.makedirs(SKIPPED_DIR, exist_ok=True)

# Storage
skipped_no_hand = 0
skipped_blurry = 0
total_images = 0
processed_images = 0
data = []
labels = []

# Extract consistent features from landmarks
def extract_landmark_features(hand_landmarks, handedness_label):
    landmarks_array = []
    
    # 1. Absolute normalized coordinates
    for landmark in hand_landmarks.landmark:
        landmarks_array.extend([landmark.x, landmark.y, landmark.z])
    
    # 2. Normalize to wrist
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    for landmark in hand_landmarks.landmark:
        landmark.x -= wrist.x
        landmark.y -= wrist.y
        landmark.z -= wrist.z
    
    # 3. Tip distances
    for tip_id in [mp_hands.HandLandmark.THUMB_TIP, 
                   mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]:
        tip = hand_landmarks.landmark[tip_id]
        distance = np.sqrt(tip.x**2 + tip.y**2 + tip.z**2)
        landmarks_array.append(distance)
    
    # 4. Angles between joints
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
        (mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_TIP),
        (mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_TIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
        (mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_TIP),
        (mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_TIP)
    ]:
        base = hand_landmarks.landmark[finger_base]
        mid = hand_landmarks.landmark[finger_mid]
        tip = hand_landmarks.landmark[finger_tip]
        angle = calculate_angle(base, mid, tip)
        landmarks_array.append(angle)

    # 5. Add handedness as the last feature
    handedness_value = 1.0 if handedness_label == 'Right' else 0.0
    landmarks_array.append(handedness_value)

    return landmarks_array


# Process each image
for dir_ in sorted(os.listdir(DATA_DIR)):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        total_images += 1
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            skipped_blurry += 1
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            skipped_no_hand += 1
            cv2.imwrite(os.path.join(SKIPPED_DIR, f"no_hand_{dir_}_{img_path}"), img)
            continue
        
        # Only access handedness after confirming we have hand landmarks
        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label
            
        hand_landmarks = results.multi_hand_landmarks[0]  # Get first detected hand
        
        # Extract features from landmarks
        fhandedness_label = 'Right'
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label

        features = extract_landmark_features(hand_landmarks, handedness_label)
        
        data.append(features)
        labels.append(dir_)
        processed_images += 1

data_array = np.array(data, dtype=np.float32)
labels_array = np.array(labels)

# Convert labels to integers
unique_labels = np.unique(labels_array)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
integer_labels = np.array([label_to_index[label] for label in labels_array], dtype=np.int32)

# Define label_mapping before using it
label_mapping = {idx: label for label, idx in label_to_index.items()}

# Now check class distribution
unique, counts = np.unique(integer_labels, return_counts=True)
print("Class distribution:")
print(dict(zip([label_mapping[i] for i in unique], counts)))

# Save mapping for later use - this line should be removed as it's redundant
# label_mapping = {idx: label for label, idx in label_to_index.items()}

try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({
            'data': data_array,
            'labels': integer_labels,
            'label_mapping': label_mapping
        }, f)
    print("✅ data.pickle created successfully!")
except Exception as e:
    print(f"❌ Error saving data.pickle: {e}")

print("\n=== 📊 Dataset Summary ===")
print(f"📸 Total images scanned: {total_images}")
print(f"✅ Successfully processed: {processed_images}")
print(f"❌ Skipped (no hand detected): {skipped_no_hand}")
print(f"⚠️ Skipped (blurry/corrupted images): {skipped_blurry}")
print(f"📂 Final dataset size: {len(data)} samples")
print(f"🏷️ Classes: {', '.join(unique_labels)}")
print("✅ Dataset created successfully!")
