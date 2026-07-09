import cv2
import mediapipe as mp
import numpy as np
import random
import urllib.request
from pathlib import Path


MAX_BLUR = 45
BLUR_STEP_UP = 3
BLUR_STEP_DOWN = 2
STABLE_POSE_FRAMES = 3
STABLE_LOVE_FRAMES = 3
LOVE_RAIN_BATCH_SIZE = 4
LOVE_RAIN_INTERVAL_FRAMES = 4
MAX_LOVE_PHOTOS = 70
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent.parent / "hand_landmarker.task"
LOVE_IMAGE_PATH = Path(__file__).with_name("Lovesign.jpeg")


def get_landmarks(hand_landmarks):
    return hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks


def is_finger_up(hand_landmarks, tip_id, pip_id):
    """Return True when a fingertip is above its PIP joint."""
    landmarks = get_landmarks(hand_landmarks)
    tip = landmarks[tip_id]
    pip = landmarks[pip_id]
    return tip.y < pip.y


def is_peace_sign(hand_landmarks):
    """Detect a flexible peace sign: index and middle up, ring and pinky down."""
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)
    ring_up = is_finger_up(hand_landmarks, 16, 14)
    pinky_up = is_finger_up(hand_landmarks, 20, 18)

    return index_up and middle_up and not ring_up and not pinky_up


def landmark_distance(hand_landmarks, first_id, second_id):
    landmarks = get_landmarks(hand_landmarks)
    first = landmarks[first_id]
    second = landmarks[second_id]
    delta_x = first.x - second.x
    delta_y = first.y - second.y
    return (delta_x ** 2 + delta_y ** 2) ** 0.5


def is_love_sign(hand_landmarks):
    """Detect finger heart or ILY-style love sign."""
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)
    ring_up = is_finger_up(hand_landmarks, 16, 14)
    pinky_up = is_finger_up(hand_landmarks, 20, 18)
    thumb_index_close = landmark_distance(hand_landmarks, 4, 8) < 0.085

    finger_heart = thumb_index_close and not middle_up and not ring_up and not pinky_up
    ily_sign = index_up and pinky_up and not middle_up and not ring_up

    return finger_heart or ily_sign


def make_odd(value):
    """Convert a value to a positive odd integer for Gaussian blur kernels."""
    kernel = max(1, int(value))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def apply_smooth_blur(frame, blur_strength):
    """Apply Gaussian blur when blur_strength is above zero."""
    if blur_strength <= 0:
        return frame

    kernel = make_odd(blur_strength)
    if kernel <= 1:
        return frame

    return cv2.GaussianBlur(frame, (kernel, kernel), 0)


def create_heart_mask(size):
    """Create a heart-shaped alpha mask."""
    t = np.linspace(0, 2 * np.pi, 220)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

    points = np.column_stack(
        (
            ((x + 17) / 34 * (size - 1)).astype(np.int32),
            ((17 - y) / 34 * (size - 1)).astype(np.int32),
        )
    )

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def create_heart_photo(image, size):
    """Resize the source photo into a heart-shaped RGBA image."""
    if image is None:
        return None

    image_height, image_width = image.shape[:2]
    scale = max(size / image_width, size / image_height)
    resized_width = int(image_width * scale)
    resized_height = int(image_height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    x1 = max(0, (resized_width - size) // 2)
    y1 = max(0, (resized_height - size) // 2)
    square = resized[y1 : y1 + size, x1 : x1 + size]

    if square.shape[0] != size or square.shape[1] != size:
        square = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

    alpha = create_heart_mask(size)
    return np.dstack((square, alpha))


def overlay_rgba(frame, image_rgba, x, y):
    """Alpha-blend an RGBA image onto a BGR frame."""
    if image_rgba is None:
        return

    frame_height, frame_width = frame.shape[:2]
    image_height, image_width = image_rgba.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_width, int(x) + image_width)
    y2 = min(frame_height, int(y) + image_height)

    if x1 >= x2 or y1 >= y2:
        return

    image_x1 = x1 - int(x)
    image_y1 = y1 - int(y)
    image_x2 = image_x1 + (x2 - x1)
    image_y2 = image_y1 + (y2 - y1)

    overlay = image_rgba[image_y1:image_y2, image_x1:image_x2]
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    frame_region = frame[y1:y2, x1:x2].astype(np.float32)
    blended = overlay[:, :, :3].astype(np.float32) * alpha + frame_region * (1 - alpha)
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)


def spawn_love_photos(particles, love_image, frame_width, frame_height):
    """Create several falling heart-photo particles."""
    if love_image is None:
        return

    for _ in range(LOVE_RAIN_BATCH_SIZE):
        size = random.randint(
            max(32, int(frame_width * 0.045)),
            max(54, int(frame_width * 0.105)),
        )
        start_y_min = -max(size, int(frame_height * 0.25))
        start_y_max = -size
        particles.append(
            {
                "x": random.uniform(-size, frame_width),
                "y": random.uniform(start_y_min, start_y_max),
                "speed": random.uniform(frame_height * 0.008, frame_height * 0.018),
                "drift": random.uniform(-frame_width * 0.003, frame_width * 0.003),
                "image": create_heart_photo(love_image, size),
            }
        )

    if len(particles) > MAX_LOVE_PHOTOS:
        del particles[: len(particles) - MAX_LOVE_PHOTOS]


def draw_love_rain(frame, particles):
    """Move and draw active falling heart photos."""
    frame_height = frame.shape[0]
    active_particles = []

    for particle in particles:
        particle["x"] += particle["drift"]
        particle["y"] += particle["speed"]
        overlay_rgba(frame, particle["image"], particle["x"], particle["y"])

        image_height = particle["image"].shape[0] if particle["image"] is not None else 0
        if particle["y"] < frame_height + image_height:
            active_particles.append(particle)

    particles[:] = active_particles


def download_model_if_needed():
    """Download the MediaPipe hand landmarker model once."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    print("Model MediaPipe belum ada. Mengunduh hand_landmarker.task...")
    temp_path = MODEL_PATH.with_suffix(".task.tmp")

    try:
        urllib.request.urlretrieve(MODEL_URL, temp_path)
        temp_path.replace(MODEL_PATH)
    except Exception as error:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            "Gagal mengunduh model MediaPipe. Pastikan koneksi internet aktif, "
            f"lalu coba jalankan lagi. URL model: {MODEL_URL}"
        ) from error

    print(f"Model berhasil disimpan di: {MODEL_PATH}")
    return MODEL_PATH


def draw_status_text(frame, pose_detected):
    status_text = "POSE 2 DETECTED - BLUR ON" if pose_detected else "NORMAL"
    status_color = (0, 255, 0) if pose_detected else (255, 255, 255)

    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Press Q to exit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    vision = mp.tasks.vision
    drawing_utils = vision.drawing_utils
    drawing_styles = vision.drawing_styles
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    try:
        model_path = download_model_if_needed()
    except RuntimeError as error:
        print(f"Error: {error}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak bisa dibuka. Pastikan webcam tersedia dan tidak sedang dipakai aplikasi lain.")
        return

    blur_strength = 0
    pose_frame_count = 0
    love_frame_count = 0
    frame_index = 0
    love_particles = []
    love_image = cv2.imread(str(LOVE_IMAGE_PATH))
    if love_image is None:
        print(f"Warning: Foto love sign tidak ditemukan di {LOVE_IMAGE_PATH}")

    try:
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Error: Frame kamera gagal dibaca. Aplikasi dihentikan.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_frame),
                )

                frame_index += 1
                results = hand_landmarker.detect_for_video(mp_image, frame_index * 33)

                raw_pose_detected = False
                raw_love_detected = False
                if results.hand_landmarks:
                    raw_pose_detected = any(
                        is_peace_sign(hand_landmarks)
                        for hand_landmarks in results.hand_landmarks
                    )
                    raw_love_detected = any(
                        is_love_sign(hand_landmarks)
                        for hand_landmarks in results.hand_landmarks
                    )

                if raw_pose_detected:
                    pose_frame_count += 1
                else:
                    pose_frame_count = 0

                if raw_love_detected:
                    love_frame_count += 1
                else:
                    love_frame_count = 0

                pose_detected = pose_frame_count >= STABLE_POSE_FRAMES
                love_detected = love_frame_count >= STABLE_LOVE_FRAMES

                if pose_detected:
                    blur_strength = min(MAX_BLUR, blur_strength + BLUR_STEP_UP)
                else:
                    blur_strength = max(0, blur_strength - BLUR_STEP_DOWN)

                display_frame = apply_smooth_blur(frame, blur_strength)
                frame_height, frame_width = display_frame.shape[:2]

                if love_detected and frame_index % LOVE_RAIN_INTERVAL_FRAMES == 0:
                    spawn_love_photos(love_particles, love_image, frame_width, frame_height)

                # Draw landmarks after blur so the hand guide remains visible.
                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        drawing_utils.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            hand_connections,
                            drawing_styles.get_default_hand_landmarks_style(),
                            drawing_styles.get_default_hand_connections_style(),
                        )

                draw_love_rain(display_frame, love_particles)

                draw_status_text(display_frame, pose_detected)
                cv2.imshow("TikTok Webcam Blur Effect", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
