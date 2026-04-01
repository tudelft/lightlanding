import cv2
import numpy as np

# Board layout (number of chessboard squares)
SQUARES_X = 5   # columns
SQUARES_Y = 5   # rows

# Physical size (in METERS — must match calibration later)
SQUARE_LENGTH = 0.0460   # 4.6 cm
MARKER_LENGTH = 0.0400   # 4.0 cm

# ArUco dictionary
ARUCO_DICT = cv2.aruco.DICT_5X5_100

# Output resolution (higher = better print quality)
DPI = 300

# Margin around the board (important for detection)
MARGIN_CM = 2.0

# Output filenames
PNG_FILE = "charuco_board.png"
PDF_FILE = "charuco_board.pdf"

# ============================================================
# CREATE BOARD
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

# ============================================================
# COMPUTE IMAGE SIZE FOR PRINTING
# ============================================================

# Convert meters → pixels
def meters_to_pixels(m, dpi):
    inches = m / 0.0254
    return int(inches * dpi)

board_width_m = SQUARES_X * SQUARE_LENGTH
board_height_m = SQUARES_Y * SQUARE_LENGTH

margin_m = MARGIN_CM / 100.0

img_width_px = meters_to_pixels(board_width_m + 2 * margin_m, DPI)
img_height_px = meters_to_pixels(board_height_m + 2 * margin_m, DPI)

# ============================================================
# DRAW BOARD
# ============================================================

img = board.generateImage((img_width_px, img_height_px))

# Save PNG
cv2.imwrite(PNG_FILE, img)
print(f"Saved PNG: {PNG_FILE}")

# ============================================================
# OPTIONAL: SAVE AS PDF (print-friendly)
# ============================================================

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.pagesizes import A3

    c = canvas.Canvas(PDF_FILE, pagesize=A3)

    # Convert board size to cm
    board_width_cm = board_width_m * 100
    board_height_cm = board_height_m * 100

    # Center on page
    page_width, page_height = A3
    x = (page_width - board_width_cm * cm) / 2
    y = (page_height - board_height_cm * cm) / 2

    c.drawImage(PNG_FILE, x, y,
                width=board_width_cm * cm,
                height=board_height_cm * cm)

    c.save()
    print(f"Saved PDF: {PDF_FILE}")

except ImportError:
    print("reportlab not installed → skipping PDF export")
