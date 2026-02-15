from layer1_extraction import extract_landmarks
from layer2_metrics import calculate_metrics
from layer3_classify import classify_features
from generation import run_generation
import json


if __name__ == "__main__":
    image_path = r"C:\Users\user\Desktop\Final_Project\test_img\F01.jpg"
    json_output_path = r"C:\Users\user\Desktop\Final_Project\face_features.json"

    try:
        # --- Extract landmarks ---
        coords, img_shape = extract_landmarks(image_path)

        # --- Save detected landmarks for accuracy checks ---
        detected_landmarks_path = r"C:\Users\user\Desktop\Final_Project\detected_landmarks.json"
        with open(detected_landmarks_path, 'w') as f:
            json.dump({"landmarks": coords}, f, indent=4)
        print(f"Detected landmarks saved at: {detected_landmarks_path}")

        # --- Calculate metrics ---
        metrics = calculate_metrics(coords, img_shape)

        # --- Classify features ---
        result_json, human_text = classify_features(metrics)

        # --- Print outputs ---
        print("JSON Output:\n", json.dumps(result_json, indent=4))
        print("\nHuman-Readable Description:\n", human_text)

        # --- Save JSON to file ---
        with open(json_output_path, 'w') as f:
            json.dump(result_json, f, indent=4)
        print(f"\nJSON file saved at: {json_output_path}")

        run_generation(
            face_features_path=json_output_path,
            knowledge_path="./knowledge",
            output_path="final_makeup_recommendations.json",
            model_name="phi3"
        )

    except Exception as e:
        print("Error:", e)
    
    
