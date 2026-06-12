import cv2 as cv

from src.modules.hand_tracker import HandTracker


def main():
    tracker = HandTracker()
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        frame = tracker.process_frame(frame)

        state = tracker.get_hand_state()

        if state is not None:
            palm_x = state["palm_x"]
            palm_y = state["palm_y"]
            is_closed = state["is_closed"]
            status = "CLOSED" if is_closed else "OPEN"
            color  = (0, 0, 255) if is_closed else (0, 255, 0)

            cv.putText(frame, f"Palm: ({palm_x:.2f}, {palm_y:.2f})", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv.putText(frame, f"Hand: {status}", (20, 75),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv.putText(frame, "No hand detected", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.imshow("Hand Tracker Test", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()