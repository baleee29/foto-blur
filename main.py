import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from pathlib import Path


MAX_BLUR = 45
BLUR_STEP_UP = 3
BLUR_STEP_DOWN = 2
STABLE_POSE_FRAMES = 3
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def is_finger_up(hand_landmarks, tip_id, pip_id):
    """Return True when a fingertip is above its PIP joint."""
    landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
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
    frame_index = 0

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
                if results.hand_landmarks:
                    raw_pose_detected = any(
                        is_peace_sign(hand_landmarks)
                        for hand_landmarks in results.hand_landmarks
                    )

                if raw_pose_detected:
                    pose_frame_count += 1
                else:
                    pose_frame_count = 0

                pose_detected = pose_frame_count >= STABLE_POSE_FRAMES

                if pose_detected:
                    blur_strength = min(MAX_BLUR, blur_strength + BLUR_STEP_UP)
                else:
                    blur_strength = max(0, blur_strength - BLUR_STEP_DOWN)

                display_frame = apply_smooth_blur(frame, blur_strength)

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

                draw_status_text(display_frame, pose_detected)
                cv2.imshow("TikTok Webcam Blur Effect", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
