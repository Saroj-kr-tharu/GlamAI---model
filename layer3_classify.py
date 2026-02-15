def classify_features(metrics):

    result = {}
    human_text = []

    # --- FACE SHAPE (Facial Index) ---
    # Facial index (height/width) is a widely used anthropometric measure
    ratio = metrics['face_ratio']
    if ratio < 0.85:
        face_shape = 'broad'  # corresponds to euryprosopic/hypereuryprosopic
        secondary = None
    elif ratio < 0.90:
        face_shape = 'round'  # mesoprosopic
        secondary = None
    elif ratio < 0.95:
        face_shape = 'oval'  # leptoprosopic
        secondary = 'round'
    elif ratio < 1.00:
        face_shape = 'long'  # hyperleptoprosopic
        secondary = 'oval'
    else:
        face_shape = 'very long'
        secondary = 'long'
    result['face_shape'] = {'primary': face_shape, 'secondary': secondary, 'ratio': ratio}
    label = f"Your face shape is {face_shape}"
    if secondary:
        label += f" with subtle {secondary} influence"
    human_text.append(label + ".")

    # --- FACE SYMMETRY ---
    # Eye alignment difference normalized; smaller is better symmetry
    sym = metrics['eye_symmetry']
    if sym < 0.015:
        symmetry = 'high'
    elif sym < 0.03:
        symmetry = 'moderate'
    else:
        symmetry = 'noticeable asymmetry'
    result['face_symmetry'] = {'level': symmetry, 'eye_alignment': sym}
    human_text.append(f"Your facial symmetry is {symmetry}.")

    # --- NOSE (Nasal index) ---
    # Width / length ratios used in anthropometry
    nose_ratio = metrics['nose_width']
    if nose_ratio < 0.14:
        nose_width = 'narrow'
    elif nose_ratio < 0.18:
        nose_width = 'average'
    else:
        nose_width = 'wide'

    nose_length = metrics['nose_length']
    if nose_length < 0.28:
        nose_length_type = 'short'
    elif nose_length < 0.36:
        nose_length_type = 'average'
    else:
        nose_length_type = 'long'

    # Tip shape derived from width and length
    if nose_length_type == 'short' and nose_width == 'narrow':
        nose_tip = 'rounded'
    elif nose_width == 'wide':
        nose_tip = 'soft curve'
    else:
        nose_tip = 'defined'

    result['nose'] = {'width': nose_width, 'length': nose_length_type, 'tip': nose_tip,
                      'metrics': {'width_ratio': nose_ratio, 'length_ratio': nose_length}}
    human_text.append(f"Your nose is {nose_width} in width, {nose_length_type} in length, with a {nose_tip} tip.")

    # --- EYES ---
    # Eye height/width ratios describe shape; inter-eye distance describes spacing
    eye_ratio = (metrics['left_eye_height'] + metrics['right_eye_height']) / \
                (metrics['left_eye_width'] + metrics['right_eye_width'])
    if eye_ratio > 0.8:
        eye_shape = 'round'
    elif eye_ratio > 0.6:
        eye_shape = 'almond'
    else:
        eye_shape = 'hooded'

    diff = metrics['left_eye_height'] - metrics['right_eye_height']
    if abs(diff) > 0.02:
        eye_orientation = 'asymmetric'
    else:
        eye_orientation = 'balanced'

    inter_eye_ratio = metrics['inter_eye_distance']
    if inter_eye_ratio < 0.32:
        eye_spacing = 'close-set'
    elif inter_eye_ratio < 0.36:
        eye_spacing = 'balanced'
    else:
        eye_spacing = 'wide-set'

    result['eyes'] = {'shape': eye_shape, 'orientation': eye_orientation, 'spacing': eye_spacing,
                      'metrics': {'eye_ratio': eye_ratio, 'inter_eye_distance': inter_eye_ratio}}
    human_text.append(f"Your eyes are {eye_shape}, {eye_orientation}, and {eye_spacing}.")

    # --- LIPS ---
    # Fullness = upper + lower height; balance = upper/lower ratio
    upper = metrics['upper_lip_height']
    lower = metrics['lower_lip_height']
    fullness_ratio = upper + lower
    if fullness_ratio < 0.05:
        lip_fullness = 'thin'
        secondary = 'medium' if fullness_ratio > 0.045 else None
    elif fullness_ratio < 0.08:
        lip_fullness = 'medium'
        secondary = 'thin' if fullness_ratio < 0.055 else 'full'
    else:
        lip_fullness = 'full'
        secondary = 'medium' if fullness_ratio < 0.09 else None

    ul_lr_ratio = upper / lower if lower != 0 else 1
    if ul_lr_ratio > 1.05:
        lip_balance = 'upper-dominant'
    elif ul_lr_ratio < 0.95:
        lip_balance = 'lower-dominant'
    else:
        lip_balance = 'balanced'

    if lip_fullness == 'full' and lip_balance == 'balanced':
        lip_contour = 'pouty'
    elif lip_fullness == 'medium' and lip_balance == 'upper-dominant':
        lip_contour = 'bow-shaped'
    else:
        lip_contour = 'natural'

    result['lips'] = {'fullness': lip_fullness, 'secondary': secondary, 'balance': lip_balance,
                      'contour': lip_contour, 'metrics': {'fullness_ratio': fullness_ratio, 'ul_lr_ratio': ul_lr_ratio}}
    lip_label = f"Your lips are {lip_fullness}"
    if secondary:
        lip_label += f" with mild {secondary} influence"
    lip_label += f", {lip_balance}, with a {lip_contour} contour."
    human_text.append(lip_label)

    # --- EYEBROWS ---
    avg_angle = (metrics['left_brow_angle'] + metrics['right_brow_angle']) / 2
    if avg_angle < 5:
        arch_type = 'straight'
    elif avg_angle < 15:
        arch_type = 'soft arch'
    else:
        arch_type = 'defined arch'
    result['eyebrows'] = {'arch': arch_type, 'thickness': 'natural', 'angle': avg_angle}
    human_text.append(f"Your eyebrows are {arch_type} with natural thickness.")

    # --- JAW & CHIN ---
    jaw_width_ratio = metrics['jaw_width']
    if jaw_width_ratio < 0.35:
        jaw_type = 'narrow'
    elif jaw_width_ratio < 0.45:
        jaw_type = 'balanced'
    else:
        jaw_type = 'wide'

    chin_proj = metrics['chin_projection']
    if chin_proj < 0.03:
        chin_shape = 'pointed'
    elif chin_proj < 0.05:
        chin_shape = 'balanced'
    else:
        chin_shape = 'prominent'

    result['jaw_chin'] = {'jaw': jaw_type, 'chin_shape': chin_shape,
                          'metrics': {'jaw_width_ratio': jaw_width_ratio, 'chin_projection': chin_proj}}
    human_text.append(f"Your jaw is {jaw_type} and your chin is {chin_shape}.")

    # --- CHEEKBONES ---
    cheek_prom = metrics['cheekbone_prominence']
    cheek_height = metrics['cheekbone_height']

    if cheek_prom < 0.8:
        cheek_prom_label = 'subtle'
    elif cheek_prom < 1.0:
        cheek_prom_label = 'moderate'
    else:
        cheek_prom_label = 'prominent'

    if cheek_height < 0.1:
        cheek_height_label = 'low-set'
    elif cheek_height < 0.2:
        cheek_height_label = 'balanced'
    else:
        cheek_height_label = 'high-set'

    result['cheekbones'] = {'prominence': cheek_prom_label, 'height': cheek_height_label,
                            'definition': 'natural', 'metrics': {'prominence': cheek_prom, 'height_ratio': cheek_height}}
    human_text.append(f"Your cheekbones are {cheek_prom_label} and {cheek_height_label}, giving your face well-structured contours.")

    return result, "\n".join(human_text)
