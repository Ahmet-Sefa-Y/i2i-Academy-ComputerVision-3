import cv2
import mediapipe as mp

# MediaPipe hand-tracking utilities and landmark indexes.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGERTIP_IDS = [4, 8, 12, 16, 20]
PIP_JOINT_IDS = [3, 6, 10, 14, 18]


def count_open_fingers(hand_landmarks, hand_label):
    """Return the number of open fingers for one detected hand."""
    landmarks = hand_landmarks.landmark
    open_fingers = 0

    # Thumb: compare horizontal position. The direction differs by hand side.
    thumb_tip_x = landmarks[FINGERTIP_IDS[0]].x
    thumb_joint_x = landmarks[PIP_JOINT_IDS[0]].x

    if hand_label == "Right":
        thumb_is_open = thumb_tip_x < thumb_joint_x
    else:
        thumb_is_open = thumb_tip_x > thumb_joint_x

    if thumb_is_open:
        open_fingers += 1

    # Index, middle, ring, and pinky: a finger is open when its tip is
    # above its middle joint in the camera frame.
    for tip_id, pip_id in zip(FINGERTIP_IDS[1:], PIP_JOINT_IDS[1:]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            open_fingers += 1

    return open_fingers


def main():
    """Open webcam, detect one hand, and show the live open-finger count."""
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            frame_ok, frame = camera.read()

            if not frame_ok:
                print("Could not read a frame from the webcam.")
                break

            # Mirror the image to make it behave like a selfie camera.
            frame = cv2.flip(frame, 1)

            # OpenCV uses BGR; MediaPipe expects RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            finger_count = 0
            status_text = "No hand detected"

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                hand_label = results.multi_handedness[0].classification[0].label

                finger_count = count_open_fingers(hand_landmarks, hand_label)
                status_text = f"Detected hand: {hand_label}"

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

            # Live overlay for the task requirement.
            cv2.rectangle(frame, (15, 15), (445, 125), (30, 30, 30), -1)
            cv2.putText(
                frame,
                f"Open fingers: {finger_count}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                status_text,
                (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("i2i Computer Vision - Finger Counter", frame)

            # Press Q or Esc to exit.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
