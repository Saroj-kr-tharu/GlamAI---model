import cv2
import numpy as np
from layer1_extraction import extract_landmarks

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_PATH = "test_img\F01.jpg" 

# -------------------------------
# 1. Run your existing landmark extractor
# -------------------------------
coords, resized_shape = extract_landmarks(
    IMAGE_PATH,
    show_steps=False,
    save_steps=False
)

h, w, _ = resized_shape

# Remove z-coordinate for accuracy computation
detected_points = [(x, y) for (x, y, z) in coords]

# Reload resized image for annotation
image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (512, 512))

# Draw detected landmarks (green)
for (x, y) in detected_points:
    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# -------------------------------
# 2. Manual ground-truth selection
# -------------------------------
manual_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Landmark Accuracy Validation", image)

cv2.imshow("Landmark Accuracy Validation", image)
cv2.setMouseCallback("Landmark Accuracy Validation", click_event)

print("\nINSTRUCTIONS:")
print("- Click on key facial points (eyes, nose tip, lips, jaw, etc.)")
print("- Green dots = detected landmarks")
print("- Red dots   = your ground-truth points")
print("- Press ESC when done\n")

while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

if len(manual_points) == 0:
    raise RuntimeError("No ground-truth points selected.")

# -------------------------------
# 3. Accuracy computation
# -------------------------------
def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

pixel_errors = []

for gt in manual_points:
    distances = [euclidean(gt, dp) for dp in detected_points]
    pixel_errors.append(min(distances))

mean_pixel_error = np.mean(pixel_errors)
max_pixel_error = np.max(pixel_errors)

# Normalized error (scale-independent, research-grade)
face_width = max([x for x, y in detected_points]) - min([x for x, y in detected_points])
normalized_error = mean_pixel_error / face_width

# -------------------------------
# 4. Results
# -------------------------------
print("\n========== LANDMARK ACCURACY REPORT ==========")
print(f"Number of manual points : {len(manual_points)}")
print(f"Average pixel error     : {mean_pixel_error:.2f} px")
print(f"Maximum pixel error     : {max_pixel_error:.2f} px")
print(f"Normalized error        : {normalized_error:.4f}")
print("============================================")
