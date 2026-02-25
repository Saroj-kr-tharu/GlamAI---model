import os
import json
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from layer1_extraction import extract_landmarks
from layer2_metrics import calculate_metrics
from layer3_classify import classify_features
from generation import run_generation

#  Config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from any frontend

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit


#  Helpers 
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


#  Routes 
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "GlamAi API is running by saroj "}), 200


@app.route("/analyze", methods=["POST"])
def analyze_face():
    """
    Accepts an image file, runs the full pipeline:
      Layer 1  → extract landmarks (MediaPipe)
      Layer 2  → calculate metrics
      Layer 3  → classify features
      Generate → RAG + LLM makeup recommendations
    Returns JSON with face features and recommendations.
    """

    #  Validate upload 
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as form-data with key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    #  Save uploaded image 
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(image_path)

    try:
        # 3. Layer 1 – Extract landmarks 
        coords, img_shape = extract_landmarks(image_path, show_steps=False, save_steps=False)

        #  Layer 2 – Calculate metrics 
        metrics = calculate_metrics(coords, img_shape)

        #  5. Layer 3 – Classify features 
        face_features, human_text = classify_features(metrics)

        # Save face features for generation step
        features_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_features.json")
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(face_features, f, indent=4)

        #  6. Generation – RAG + LLM recommendations 
        recommendations = run_generation(
            face_features_path=features_path,
            knowledge_path="./knowledge",
            output_path=os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_recommendations.json"),
            model_name="phi3"
        )

        #  7. Build response 
        response = {
            "success": True,
            "face_features": face_features,
            "human_readable": human_text,
            "recommendations": recommendations,
        }

        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred.", "details": str(e)}), 500

    finally:
        # Clean up uploaded image
        if os.path.exists(image_path):
            os.remove(image_path)


#  Run 
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
