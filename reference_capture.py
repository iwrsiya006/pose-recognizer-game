import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json

# Setup MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Create folder for references
os.makedirs("references", exist_ok=True)

# Leaderboard file
LEADERBOARD_FILE = "leaderboard.json"

# Load or create leaderboard
def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []

def save_leaderboard(data):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f)

# Function to show leaderboard graphically in OpenCV window
def show_leaderboard():
    leaderboard = load_leaderboard()
    board_img = np.zeros((500, 600, 3), dtype=np.uint8)

    # Title
    cv2.putText(board_img, "LEADERBOARD", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    # Show top 5 scores
    for i, entry in enumerate(leaderboard[:5], start=1):
        text = f"{i}. {entry['name']} - {entry['score']}"
        cv2.putText(board_img, text, (100, 100 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Leaderboard", board_img)
    cv2.waitKey(5000)  # Display for 5 seconds
    cv2.destroyWindow("Leaderboard")

# Capture reference poses for 5 levels
cap = cv2.VideoCapture(0)

for level in range(1, 6):
    print(f"\nGet ready for Level {level} reference pose. You have 10 seconds.")
    start_time = time.time()
    saved = False

    # Indicator popup for new level
    indicator_img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(indicator_img, f"Level {level} Reference", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow("Level Indicator", indicator_img)
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyWindow("Level Indicator")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip for natural view
        frame = cv2.flip(frame, 1)
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show countdown
        elapsed = int(time.time() - start_time)
        remaining = 10 - elapsed
        cv2.putText(frame, f"Level {level} | Time left: {remaining}s", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Reference Capture", frame)

        # After 10s, save pose landmarks
        if elapsed >= 10 and result.pose_landmarks and not saved:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
            np.save(f"references/level{level}.npy", landmarks)
            print(f"Reference for Level {level} saved!")
            saved = True
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("All references captured!")
cap.release()
cv2.destroyAllWindows()

# Show graphical leaderboard after capturing references
show_leaderboard()
