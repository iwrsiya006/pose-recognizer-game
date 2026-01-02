import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import sys
from PIL import ImageFont, ImageDraw, Image
import pygame

# Initialize pygame mixer for audio
pygame.mixer.init()

# Ensure script uses the current python environment
if sys.executable is None:
    print("[ERROR] Could not detect Python executable. Please run this script using: python3 pose_game.py")
    sys.exit(1)

# Setup MediaPipe Pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Leaderboard file for saving top scores
LEADERBOARD_FILE = "leaderboard.json"

# Path to Star Wars font installed on your Mac
FONT_PATH = "/Users/aiclubsru/Desktop/pyclub/pose_game/fonts/DeathStar-VmWB.ttf" #'/Volumes/T7 Shield/pyclub/pose_game/fonts/DeathStar-VmWB.ttf'

# Audio paths
AUDIO_PATH = "/Users/aiclubsru/Desktop/pyclub/pose_game/audiogame" #'/Volumes/T7 Shield/pyclub/pose_game/audiogame'
OPENING_SOUND = os.path.join(AUDIO_PATH, "opening.mp3")
LEVEL_SOUND = os.path.join(AUDIO_PATH, "level.mp3")
GAMEOVER_SOUND = os.path.join(AUDIO_PATH, "gameover.mp3")
WINNER_SOUND = os.path.join(AUDIO_PATH, "winner.mp3")

# Game resolution
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Scale factor for dynamic text sizes based on resolution
SCALE_X = WINDOW_WIDTH / 1280
SCALE_Y = WINDOW_HEIGHT / 720
SCALE = min(SCALE_X, SCALE_Y)

# Try loading font once, fallback if not available
try:
    BASE_FONT = ImageFont.truetype(FONT_PATH, int(40 * SCALE))
except OSError:
    print(f"[WARNING] Could not load font at {FONT_PATH}, falling back to default.")
    BASE_FONT = ImageFont.load_default()

# Play sound helper
def play_sound(sound_path):
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[WARNING] Could not play sound {sound_path}: {e}")

# Load leaderboard from file, or return empty if none exists
def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []

# Save leaderboard data back to file
def save_leaderboard(data):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f)

# Function to reset leaderboard data
def reset_leaderboard():
    save_leaderboard([])

# Function to calculate similarity between two poses
def pose_similarity(ref, user):
    if ref.shape != user.shape:
        return 0

    ref_norm = ref - np.mean(ref, axis=0)
    user_norm = user - np.mean(user, axis=0)

    ref_vec = ref_norm.flatten()
    user_vec = user_norm.flatten()

    cos_sim = np.dot(ref_vec, user_vec) / (np.linalg.norm(ref_vec) * np.linalg.norm(user_vec))
    return cos_sim

# Helper function to draw text with Star Wars font and auto-scaling
def draw_text(frame, text, position, font_size=40, color=(255, 255, 255), max_width=None):
    font_size = int(font_size * SCALE)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        font = BASE_FONT

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Auto-scale font if text is too wide
    if max_width:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        while text_width > max_width and font_size > 10:
            font_size -= 2
            try:
                font = ImageFont.truetype(FONT_PATH, font_size)
            except OSError:
                font = BASE_FONT
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Function to draw leaderboard overlay on the game window
def draw_leaderboard_overlay(frame):
    leaderboard = load_leaderboard()
    start_x = frame.shape[1] - int(320 * SCALE_X)
    start_y = int(40 * SCALE_Y)

    frame = draw_text(frame, "Leaderboard", (start_x, start_y), font_size=32, color=(0, 255, 255))
    frame = draw_text(frame, "Name    Score", (start_x, start_y + int(40 * SCALE_Y)), font_size=26, color=(255, 255, 255))

    for i, entry in enumerate(leaderboard[:5], start=1):
        name = entry['name']
        score = str(entry['score'])
        frame = draw_text(frame, f"{name:<8}{score:>6}", (start_x, start_y + int(40 * SCALE_Y) + i * int(35 * SCALE_Y)), font_size=24, color=(200, 200, 200), max_width=int(300 * SCALE_X))
    return frame

# Function to capture name input directly in the game window
def get_name_input():
    play_sound(OPENING_SOUND)
    name = ""
    cv2.namedWindow("Game", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Game", WINDOW_WIDTH, WINDOW_HEIGHT)

    while True:
        frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        frame = draw_text(frame, "Enter Your Name:", (200, 250), font_size=70, color=(0, 255, 255))
        frame = draw_text(frame, name, (200, 450), font_size=90, color=(255, 255, 255), max_width=900)
        cv2.imshow("Game", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            return name if name else "Player"
        elif key in [8, 127]:  # Backspace or Delete (works on Mac too)
            if len(name) > 0:
                name = name[:-1]
        elif 32 <= key <= 126:  # Printable characters
            name += chr(key)

# Function to draw game over screen with leaderboard
def show_game_over_screen():
    play_sound(GAMEOVER_SOUND)
    over_img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    over_img = draw_text(over_img, "GAME OVER", (WINDOW_WIDTH//2 - int(325 * SCALE_X), int(120 * SCALE_Y)), font_size=110, color=(255, 0, 0))

    # Vertical button list
    options = ["R = Restart", "N = New Game", "Q = Quit", "L = Reset Leaderboard"]
    for i, text in enumerate(options):
        over_img = draw_text(over_img, text, (150, 350 + i * int(70 * SCALE_Y)), font_size=30, color=(0, 255, 255))

    # Leaderboard display positioned clearly on right side inside screen
    leaderboard = load_leaderboard()
    over_img = draw_text(over_img, "Leaderboard", (WINDOW_WIDTH - int(460 * SCALE_X), int(250 * SCALE_Y)), font_size=46, color=(0, 255, 255))
    over_img = draw_text(over_img, "Name    Score", (WINDOW_WIDTH - int(460 * SCALE_X), int(300 * SCALE_Y)), font_size=34, color=(255, 255, 255))

    for i, entry in enumerate(leaderboard[:5], start=1):
        name = entry['name']
        score = str(entry['score'])
        over_img = draw_text(over_img, f"{name:<8}{score:>6}", (WINDOW_WIDTH - int(460 * SCALE_X), int(300 * SCALE_Y) + i * int(40 * SCALE_Y)), font_size=30, color=(200, 200, 200), max_width=int(300 * SCALE_X))

    cv2.imshow("Game", over_img)

# Main game loop for one player
def play_game(player_name):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Game", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Game", WINDOW_WIDTH, WINDOW_HEIGHT)

    score = 0
    game_over = False

    for level in range(1, 6):
        ref_path = f"references/level{level}.npy"
        if not os.path.exists(ref_path):
            print(f"Missing reference for Level {level}!")
            break
        reference_pose = np.load(ref_path)

        # Display reference text for 3 seconds (centered)
        play_sound(LEVEL_SOUND)
        ref_img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        ref_img = draw_text(ref_img, f" Level {level}", (WINDOW_WIDTH//2 - int(250 * SCALE_X), WINDOW_HEIGHT//2 - int(50 * SCALE_Y)), font_size=90, color=(255, 255, 255))
        cv2.imshow("Game", ref_img)
        cv2.waitKey(3000)

        start_time = time.time()
        captured_pose = None

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                elapsed = int(time.time() - start_time)
                remaining = 10 - elapsed

                frame = draw_text(frame, f"Level {level}", (30, 50), font_size=50, color=(0, 255, 0))
                frame = draw_text(frame, f"{remaining}s", (WINDOW_WIDTH - int(200 * SCALE_X), WINDOW_HEIGHT - int(80 * SCALE_Y)), font_size=80, color=(0, 255, 255))

                if elapsed >= 10:
                    captured_pose = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
                    break

            # Show keybinds
            frame = draw_text(frame, "R=Restart  N=New Game  Q=Quit", (30, WINDOW_HEIGHT - int(40 * SCALE_Y)), font_size=34, color=(0, 255, 255))

            # Draw leaderboard overlay
            frame = draw_leaderboard_overlay(frame)

            cv2.imshow("Game", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, True
            elif key == ord('r'):
                cap.release()
                cv2.destroyAllWindows()
                return player_name, False
            elif key == ord('n'):
                cap.release()
                cv2.destroyAllWindows()
                return get_name_input(), False

        if captured_pose is not None:
            similarity = pose_similarity(reference_pose, captured_pose)

            if similarity > 0.85:
                score = level
                continue
            else:
                game_over = True
                break

    cap.release()
    cv2.destroyAllWindows()

    # Update leaderboard
    leaderboard = load_leaderboard()
    leaderboard.append({"name": player_name, "score": score})
    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)[:5]
    save_leaderboard(leaderboard)

    # Show game over window if failed
    if game_over:
        show_game_over_screen()
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                return player_name, False
            elif key == ord('n'):
                return get_name_input(), False
            elif key == ord('q'):
                return None, True
            elif key == ord('l'):
                reset_leaderboard()
                show_game_over_screen()

    if score == 5:
        play_sound(WINNER_SOUND)
        win_img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        win_img = draw_text(win_img, "WINNER!", (WINDOW_WIDTH//2 - int(250 * SCALE_X), int(120 * SCALE_Y)), font_size=110, color=(0, 255, 0))

        # Leaderboard under winner text (centered horizontally)
        leaderboard = load_leaderboard()
        lb_start_x = WINDOW_WIDTH//2 - int(200 * SCALE_X)
        win_img = draw_text(win_img, "Leaderboard", (lb_start_x, 300), font_size=46, color=(0, 255, 255))
        win_img = draw_text(win_img, "Name    Score", (lb_start_x, 350), font_size=34, color=(255, 255, 255))

        for i, entry in enumerate(leaderboard[:5], start=1):
            name = entry['name']
            score = str(entry['score'])
            win_img = draw_text(win_img, f"{name:<8}{score:>6}", (lb_start_x, 350 + i * int(40 * SCALE_Y)), font_size=30, color=(200, 200, 200), max_width=int(500 * SCALE_X))

        # Horizontal buttons below leaderboard
        options = ["N = New Game"]
        btn_y = 350 + (len(leaderboard[:5]) + 2) * int(40 * SCALE_Y)
        spacing = 300
        btn_x_start = WINDOW_WIDTH//2 - (len(options) - 1) * spacing // 2

        for i, text in enumerate(options):
            win_img = draw_text(win_img, text, (btn_x_start + i * spacing, btn_y), font_size=45, color=(0, 255, 255))

        cv2.imshow("Game", win_img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                return player_name, False
            elif key == ord('n'):
                return get_name_input(), False
            elif key == ord('q'):
                return None, True

# Game loop
if __name__ == "__main__":
    current_name = get_name_input()
    exit_game = False
    while not exit_game:
        current_name, exit_game = play_game(current_name)
