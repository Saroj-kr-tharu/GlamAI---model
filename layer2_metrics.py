import math

def calculate_metrics(coords, resized):

    h, w, _ = resized
    metrics = {}

    # --- Face Dimensions ---
    x_coords = [x for x, y, z in coords]
    y_coords = [y for x, y, z in coords]
    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)
    metrics['face_width'] = face_width
    metrics['face_height'] = face_height
    # Facial index = height / width, used in anthropometry
    metrics['face_ratio'] = face_height / face_width

    # --- Eyes ---
    left_eye_center = coords[33]   # approximate left eye center
    right_eye_center = coords[263] # approximate right eye center

    # Eye spacing (interocular distance) normalized by face width
    metrics['inter_eye_distance'] = abs(right_eye_center[0] - left_eye_center[0]) / face_width

    # Eye symmetry: vertical alignment difference normalized by face height
    metrics['eye_symmetry'] = abs(left_eye_center[1] - right_eye_center[1]) / face_height

    # Left eye width & height
    metrics['left_eye_width'] = abs(coords[133][0] - coords[173][0]) / face_width
    metrics['left_eye_height'] = abs(coords[159][1] - coords[145][1]) / face_height

    # Right eye width & height
    metrics['right_eye_width'] = abs(coords[362][0] - coords[386][0]) / face_width
    metrics['right_eye_height'] = abs(coords[386][1] - coords[374][1]) / face_height

    # --- Nose ---
    nose_tip = coords[1]
    nose_bridge = coords[168]
    nose_left = coords[98]
    nose_right = coords[327]
    # Nasal index: width / length ratios normalized
    metrics['nose_width'] = abs(nose_right[0] - nose_left[0]) / face_width
    metrics['nose_length'] = abs(nose_tip[1] - nose_bridge[1]) / face_height

    # --- Lips ---
    upper_lip = coords[13]
    lower_lip = coords[14]
    lip_left = coords[61]
    lip_right = coords[291]
    # Lip height: upper + lower relative to face height
    metrics['upper_lip_height'] = abs(upper_lip[1] - lower_lip[1]) / face_height
    metrics['lower_lip_height'] = abs(lower_lip[1] - upper_lip[1]) / face_height
    # Lip width normalized to face width
    metrics['lip_width'] = abs(lip_right[0] - lip_left[0]) / face_width

    # --- Eyebrows ---
    left_brow_inner = coords[105]
    left_brow_outer = coords[65]
    right_brow_inner = coords[334]
    right_brow_outer = coords[295]

    # Slope-based eyebrow angle in degrees
    metrics['left_brow_angle'] = math.degrees(math.atan2(
        left_brow_outer[1] - left_brow_inner[1],
        left_brow_outer[0] - left_brow_inner[0]
    ))
    metrics['right_brow_angle'] = math.degrees(math.atan2(
        right_brow_outer[1] - right_brow_inner[1],
        right_brow_outer[0] - right_brow_inner[0]
    ))

    # --- Jaw & Chin ---
    jaw_left = coords[234]
    jaw_right = coords[454]
    chin = coords[152]
    # Jaw width / face width ratio
    metrics['jaw_width'] = abs(jaw_right[0] - jaw_left[0]) / face_width
    # Chin projection relative to face height (distance from top of face)
    metrics['chin_projection'] = (chin[1] - min(y_coords)) / face_height

    # --- Cheekbones ---
    left_cheek = coords[234]
    right_cheek = coords[454]
    cheek_top = coords[10]  # approximate top of cheek
    # Cheekbone prominence = horizontal distance normalized by face width
    metrics['cheekbone_prominence'] = abs(right_cheek[0] - left_cheek[0]) / face_width
    # Cheekbone height = vertical distance from chin to cheek top normalized by face height
    metrics['cheekbone_height'] = abs(cheek_top[1] - chin[1]) / face_height

    return metrics
