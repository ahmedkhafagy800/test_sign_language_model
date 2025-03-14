import os
import cv2
import numpy as np
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Constants
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30


def setup_directories():
    """Create directories for storing collected data."""
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
            except Exception as e:
                print(f"Error creating directory for {action}/{sequence}: {e}")


def collect_data():
    """Collect keypoints from webcam feed and save them as numpy arrays with flipped camera."""
    setup_directories()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame.")
                        break

                    # Flip the frame horizontally
                    frame = cv2.flip(frame, 1)

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting {action} Video {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting {action} Video {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed.")


if __name__ == "__main__":
    collect_data()