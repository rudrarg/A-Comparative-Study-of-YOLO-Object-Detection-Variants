from ultralytics import YOLO
import cv2

# Load a more accurate YOLOv8 model
model = YOLO('yolov8m.pt')  # Use yolov8n/s/m/l/x.pt depending on speed vs accuracy

# Open webcam with DirectShow backend (for Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Optional: Resize frame to improve FPS and performance
    frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 inference
    results = model(frame)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Show object names with confidence in console (optional)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]
        print(f"{name}: {conf:.2f}")

    # Display the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()
