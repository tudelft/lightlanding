import cv2

# Select ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Show result
    cv2.imshow("Aruco Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
