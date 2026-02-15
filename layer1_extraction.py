import cv2
import mediapipe as mp
import os

# Path to the FaceLandmarker model (bundled alongside this script)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def extract_landmarks(image_path, show_steps=True, save_steps=True):
    print("[INFO] Loading image...")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError("Image not found")

    # Create output folder for saving images
    output_dir = "preprocessing_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Original image
    original = image_bgr.copy()
    if save_steps:
        cv2.imwrite(f"{output_dir}/1_original.jpg", original)

    # Resize (standardization)
    resized = cv2.resize(original, (512, 512))
    if save_steps:
        cv2.imwrite(f"{output_dir}/2_resized.jpg", resized)

    # Convert to RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if save_steps:
        cv2.imwrite(f"{output_dir}/3_rgb.jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    # Initialize MediaPipe FaceLandmarker (Tasks API – mediapipe ≥ 0.10.30)
    print("[INFO] Initializing MediaPipe FaceLandmarker...")
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"FaceLandmarker model not found at {_MODEL_PATH}. "
            "Download it from https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    print("[INFO] Running face landmark detection...")
    # FaceLandmarker expects an mp.Image in RGB format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)
    landmarker.close()

    if not result.face_landmarks:
        raise ValueError("No face detected")

    # Extract ALL landmarks (same tuple format as before: (x_px, y_px, z))
    h, w, _ = resized.shape
    face_lms = result.face_landmarks[0]
    coords = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in face_lms]
    print(f"[INFO] {len(coords)} landmarks extracted")

    # Draw mesh and highlight all coordinates
    annotated = resized.copy()
    connections = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    for conn in connections:
        x1, y1, _ = coords[conn.start]
        x2, y2, _ = coords[conn.end]
        cv2.line(annotated, (x1, y1), (x2, y2), (192, 192, 192), 1)

    # Highlight all landmarks (small green dots)
    for idx, (x, y, _) in enumerate(coords):
        cv2.circle(annotated, (x, y), 1, (0, 255, 0), -1)
        if idx % 25 == 0:
            cv2.putText(
                annotated,
                str(idx),
                (x + 2, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 255, 255),
                1
            )

    if save_steps:
        cv2.imwrite(f"{output_dir}/4_face_mesh_landmarks.jpg", annotated)

    # Show preprocessing stages
    if show_steps:
        cv2.imshow("1. Original Image", original)
        cv2.imshow("2. Resized Image", resized)
        cv2.imshow("3. RGB Converted Image", rgb_image)
        cv2.imshow("4. Face Mesh & All Landmarks", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("[INFO] Preprocessing & landmark extraction complete.")
    return coords, resized.shape
