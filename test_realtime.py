import cv2
import numpy as np
import mediapipe as mp

# Try importing TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError as e:
    print(f"Error importing TensorFlow modules: {e}")
    print("Please ensure TensorFlow is installed correctly (e.g., `pip install tensorflow`)")
    exit(1)

from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, prob_viz

# Constants
actions = np.array(['hello', 'thanks', 'iloveyou'])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
threshold = 0.8


def test_realtime():
    """Test the trained model in real-time, displaying only the latest predicted action."""
    try:
        model = load_model('action.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model 'action.h5': {e}")
        print("Ensure the model file exists and was trained using 'train_predict.py'.")
        return

    sequence = []
    current_action = ""  # Store only the latest predicted action
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Flip the frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)

            # Make detections on the flipped frame
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                confidence = res[np.argmax(res)]

                # Update current_action only if confidence exceeds threshold
                if confidence > threshold:
                    current_action = predicted_action

                # Visualize probabilities
                image = prob_viz(res, actions, image, colors)

            # Display only the latest action
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, current_action, (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Real-Time Action Recognition', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-Time testing stopped.")


if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    test_realtime()