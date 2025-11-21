import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

X = data_dict['data']
y = data_dict['labels']
label_mapping = data_dict['label_mapping']
print(f"Feature size (should include handedness): {X.shape[1]}")  # Expecting 74
print("Sample handedness values (last feature):", X[:5, -1])

# Verify that our data has the expected classes and distribution
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")
print(f"Label mapping: {label_mapping}")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip([label_mapping[i] for i in unique], counts))
print(f"Class distribution: {class_distribution}")

# Print min and max samples per class
min_samples = min(counts)
max_samples = max(counts)
print(f"Min samples per class: {min_samples}")
print(f"Max samples per class: {max_samples}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify that both training and test sets have all classes
print(f"Training set classes: {np.unique(y_train)}")
print(f"Test set classes: {np.unique(y_test)}")

# Get input shape and number of classes
input_shape = X_train.shape[1]
num_classes = len(np.unique(y))
print(f"Input shape: {input_shape}")
print(f"Number of classes: {num_classes}")

# Create a model optimized for landmark data
def create_model():
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build and train the model
model = create_model()
model.summary()  # Print model structure to verify output layer has correct units

# Add callbacks for better training
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('training_history.png')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}")

# Check predictions on test set
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[label_mapping[i] for i in unique]))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Check that all classes are being predicted
predicted_classes = np.unique(y_pred)
print(f"\nPredicted classes: {predicted_classes}")
print(f"Number of predicted classes: {len(predicted_classes)}")

if len(predicted_classes) < num_classes:
    print("WARNING: Not all classes are being predicted!")
    missing_classes = set(range(num_classes)) - set(predicted_classes)
    print(f"Missing classes: {missing_classes}")

# Save the model in TensorFlow SavedModel format
model.save('sign_language_model')

# Verify model signature before conversion
loaded_model = tf.keras.models.load_model('sign_language_model')
print("\nModel signature:")
print(f"Input shape: {loaded_model.input_shape}")
print(f"Output shape: {loaded_model.output_shape}")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model('sign_language_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# Add representative dataset for quantization (optional but recommended)
def representative_dataset():
    for i in range(min(100, len(X_test))):
        yield [np.array([X_test[i]], dtype=np.float32)]

# Enable full integer quantization
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.float16]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Test the TFLite model to verify it works
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nTFLite Model Details:")
print(f"Input: {input_details}")
print(f"Output: {output_details}")

# Test on a few samples
correct = 0
for i in range(min(10, len(X_test))):
    input_data = np.array([X_test[i]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    actual_class = y_test[i]
    print(f"Sample {i}: Predicted={predicted_class} ({label_mapping[predicted_class]}), Actual={actual_class} ({label_mapping[actual_class]})")
    if predicted_class == actual_class:
        correct += 1

print(f"\nTFLite Accuracy on test samples: {correct/min(10, len(X_test)):.2f}")

# Save the label mapping for use in Flutter
import json
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)

print("✅ TFLite model created successfully!")