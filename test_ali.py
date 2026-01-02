import time
import cv2
from camera import ImageCapture

camera = ImageCapture(1, 1280, 960)
camera.open()

print("=" * 50)
print("Camera Capture Test")
print("=" * 50)
print("Press ENTER to capture a new frame")
print("Type 'q' + ENTER to quit")
print("=" * 50)

capture_count = 0

while True:
    user_input = input("\n> Press ENTER to capture (or 'q' to quit): ")

    if user_input.lower() == 'q':
        print("Quitting...")
        break

    # Time the capture
    t0 = time.time()
    frame = camera.capture_single()
    capture_time = time.time() - t0

    capture_count += 1

    if frame is not None:
        print(f"[LOG] Capture #{capture_count}: {int(capture_time * 1000)}ms")
        print(f"[LOG] Frame size: {frame.shape[1]}x{frame.shape[0]}")

        # Show frame
        cv2.imshow("Test Capture", frame)
        cv2.waitKey(1)  # Brief wait to update window
    else:
        print(f"[LOG] Capture #{capture_count}: FAILED")

cv2.destroyAllWindows()
camera.release()
print("Done!")