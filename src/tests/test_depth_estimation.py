from ..modules.depth_estimator import DepthEstimator
import cv2 as cv

def main():
    estimator = DepthEstimator()
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        depth = estimator.estimate(frame)

        if depth is not None:
            text = f"Depth: {depth:.3f} m"
            color = (0, 255, 0)
        else:
            text = "No tag detected"
            color = (0, 0, 255)

        cv.putText(frame, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv.imshow("Depth Estimator Test", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()