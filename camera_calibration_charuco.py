#!/usr/bin/env python3
import cv2
import numpy as np
import time
from pathlib import Path

USE_FISHEYE = False
# ============================================================
# USER SETTINGS
# ============================================================

# ChArUco board definition:
# squaresX, squaresY = number of chessboard squares (not inner corners)
SQUARES_X = 5
SQUARES_Y = 5

# Physical sizes in METERS
# Example: 24 mm squares, 18 mm markers
SQUARE_LENGTH = 0.046
MARKER_LENGTH = 0.041

# ArUco dictionary
ARUCO_DICT = cv2.aruco.DICT_5X5_100

# How many good captures you want before calibrating
MIN_CAPTURES = 20

# Minimum detected ChArUco corners to accept a frame
MIN_CHARUCO_CORNERS = 8

# Save file
OUTPUT_FILE = "imx296_calibration.npz"

# ============================================================
# BOARD + DETECTORS
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
charuco_detector = cv2.aruco.CharucoDetector(board)

# 3D chessboard corner locations for the whole board
# Shape: (num_corners, 3)
board_chessboard_corners = board.getChessboardCorners()

# ============================================================
# CAMERA INPUT
# ============================================================

class CameraSource:
    def __init__(self):
        self.use_picamera2 = False
        self.picam2 = None
        self.cap = None

        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (1456, 1088), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1.0)
            self.use_picamera2 = True
            print("Using Picamera2")
        except Exception as e:
            print(f"Picamera2 not available ({e}), falling back to cv2.VideoCapture(0)")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera with Picamera2 or VideoCapture(0)")

    def read(self):
        if self.use_picamera2:
            frame = self.picam2.capture_array()
            # Picamera2 in RGB888 -> convert to BGR for OpenCV display consistency
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        else:
            return self.cap.read()

    def release(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()

# ============================================================
# HELPERS
# ============================================================

def detect_charuco(gray):
    """
    Returns:
        charuco_corners: (N,1,2) float32 or None
        charuco_ids:     (N,1) int32 or None
        vis_markers:     visualization helper
    """
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    # Detect ChArUco corners from the image + board
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    return charuco_corners, charuco_ids, vis

def collect_correspondences(charuco_corners, charuco_ids):
    """
    Convert detected ChArUco corners to object/image point pairs
    suitable for cv2.calibrateCamera().
    """
    ids = charuco_ids.flatten()
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    object_points = board_chessboard_corners[ids].astype(np.float32)
    return object_points, image_points

def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_err = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        total_err += err * err
        total_points += len(objpoints[i])

    return np.sqrt(total_err / total_points) if total_points > 0 else float("inf")

# ============================================================
# MAIN
# ============================================================

def main():
    cam = CameraSource()

    all_object_points = []
    all_image_points = []
    image_size = None

    print("\nControls:")
    print("  c = capture current frame for calibration")
    print("  r = run calibration")
    print("  q = quit\n")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]  # (width, height)

            charuco_corners, charuco_ids, vis = detect_charuco(gray)

            good = (
                charuco_ids is not None and
                len(charuco_ids) >= MIN_CHARUCO_CORNERS
            )

            status = f"captures={len(all_object_points)}"
            if good:
                status += f" | detected corners={len(charuco_ids)} | READY"
            else:
                detected = 0 if charuco_ids is None else len(charuco_ids)
                status += f" | detected corners={detected} | need >= {MIN_CHARUCO_CORNERS}"

            cv2.putText(
                vis, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if good else (0, 0, 255), 2
            )
            cv2.putText(
                vis, "c=capture  r=calibrate  q=quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

            cv2.imshow("IMX296 ChArUco Calibration", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if not good:
                    print("Not enough ChArUco corners detected in this frame.")
                    continue

                objp, imgp = collect_correspondences(charuco_corners, charuco_ids)
                all_object_points.append(objp)
                all_image_points.append(imgp)
                print(f"Captured frame {len(all_object_points)} with {len(imgp)} corners.")

            elif key == ord("r"):
            
                if len(all_object_points) < MIN_CAPTURES:
                    print(
                        f"Need at least {MIN_CAPTURES} good captures. "
                        f"Currently: {len(all_object_points)}"
                    )
                    continue
            
                # =====================================================
                # FISHEYE CALIBRATION
                # =====================================================
                if USE_FISHEYE:
            
                    print("Running fisheye calibration...")
            
                    objpoints = [
                        op.reshape(-1, 1, 3).astype(np.float64)
                        for op in all_object_points
                    ]
            
                    imgpoints = [
                        ip.reshape(-1, 1, 2).astype(np.float64)
                        for ip in all_image_points
                    ]
            
                    K = np.zeros((3, 3))
                    D = np.zeros((4, 1))
            
                    flags = (
                        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                        | cv2.fisheye.CALIB_CHECK_COND
                        | cv2.fisheye.CALIB_FIX_SKEW
                    )
            
                    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                        objpoints,
                        imgpoints,
                        image_size,
                        K,
                        D,
                        None,
                        None,
                        flags,
                        (
                            cv2.TERM_CRITERIA_EPS
                            + cv2.TERM_CRITERIA_MAX_ITER,
                            100,
                            1e-6,
                        ),
                    )
            
                    # Reprojection error
                    total_err = 0.0
                    total_points = 0
            
                    for i in range(len(objpoints)):
            
                        projected, _ = cv2.fisheye.projectPoints(
                            objpoints[i],
                            rvecs[i],
                            tvecs[i],
                            K,
                            D,
                        )
            
                        err = cv2.norm(
                            imgpoints[i],
                            projected,
                            cv2.NORM_L2,
                        )
            
                        total_err += err * err
                        total_points += len(objpoints[i])
            
                    reproj = np.sqrt(total_err / total_points)
            
                    camera_matrix = K
                    dist_coeffs = D
            
                # =====================================================
                # PERSPECTIVE / PINHOLE CALIBRATION
                # =====================================================
                else:
                    print("Running perspective calibration...")
            
                    flags = cv2.CALIB_RATIONAL_MODEL
            
                    rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
                        cv2.calibrateCamera(
                            all_object_points,
                            all_image_points,
                            image_size,
                            None,
                            None,
                            flags=flags,
                        )
                    )
            
                    reproj = compute_reprojection_error(
                        all_object_points,
                        all_image_points,
                        rvecs,
                        tvecs,
                        camera_matrix,
                        dist_coeffs,
                    )
            
                # =====================================================
                # SAVE RESULTS
                # =====================================================
            
                np.savez(
                    OUTPUT_FILE,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    image_width=image_size[0],
                    image_height=image_size[1],
                    rms=rms,
                    reprojection_error=reproj,
                    calibration_model=(
                        "fisheye" if USE_FISHEYE else "perspective"
                    ),
                )
            
                print()
                print("Calibration finished")
                print(
                    "Model:",
                    "fisheye" if USE_FISHEYE else "perspective",
                )
                print("RMS:", rms)
                print("Reprojection error:", reproj)
                print("Camera matrix:\n", camera_matrix)
                print("Distortion coeffs:\n", dist_coeffs.ravel())
                print(f"Saved to {OUTPUT_FILE}")
                print()

            elif key == ord("q"):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
