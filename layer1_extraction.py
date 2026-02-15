import cv2
import mediapipe as mp
import os

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

    # Initialize MediaPipe Face Mesh
    print("[INFO] Initializing MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    print("[INFO] Running face landmark detection...")
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        raise ValueError("No face detected")

    # Extract ALL landmarks
    h, w, _ = resized.shape
    face_landmarks = results.multi_face_landmarks[0].landmark
    coords = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in face_landmarks]
    print(f"[INFO] {len(coords)} landmarks extracted")

    # Draw mesh and highlight all coordinates
    annotated = resized.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=results.multi_face_landmarks[0],
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

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
