import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# ChromaDB setup
# -----------------------------
client = chromadb.PersistentClient()

try:
    client.delete_collection("features")
except Exception:
    pass

collection = client.get_or_create_collection(name="features")

# -----------------------------
# Helpers
# -----------------------------
def normalize(text):
    return text.strip().lower().replace(" ", "_") if isinstance(text, str) else ""

# -----------------------------
# Knowledge indexing
# -----------------------------
def index_knowledge(json_folder="./knowledge"):
    for filename in os.listdir(json_folder):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(json_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        for i, item in enumerate(data):
            feature = normalize(item.get("feature"))
            variant = normalize(item.get("variant"))
            steps = item.get("steps")

            if not feature or not variant or not steps:
                continue

            doc_id = f"{feature}_{variant}_{i}"

            collection.add(
                documents=[" ".join(steps)],
                metadatas=[{
                    "id": doc_id,
                    "feature": feature,
                    "variant": variant,
                    "technique": item.get("technique", ""),
                    "steps": json.dumps(steps, ensure_ascii=False)
                }],
                ids=[doc_id]
            )

# -----------------------------
# Load face features
# -----------------------------
def load_face_features(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Build retrieval intents
# -----------------------------
def build_feature_queries(face_features):
    intents = []

    # Nose
    nose = face_features.get("nose", {})
    if nose.get("tip"):
        intents.append({
            "feature": "nose",
            "variant": normalize(nose["tip"]),
            "query": f"{nose['tip']} nose contour technique"
        })

    # Eyes
    eyes = face_features.get("eyes", {})
    if eyes.get("shape"):
        intents.append({
            "feature": "eyes",
            "variant": normalize(eyes["shape"]),
            "query": f"{eyes['shape']} eyes makeup technique"
        })

    # Face shape
    face = face_features.get("face_shape", {})
    if face.get("primary"):
        intents.append({
            "feature": "face_shape",
            "variant": normalize(face["primary"]),
            "query": f"{face['primary']} face makeup technique"
        })

    # Lips
    lips = face_features.get("lips", {})
    if lips.get("fullness"):
        intents.append({
            "feature": "lips",
            "variant": normalize(lips["fullness"]),
            "query": f"{lips['fullness']} lips makeup technique"
        })

    # Brows
    brows = face_features.get("eyebrows", {})
    if brows.get("arch"):
        intents.append({
            "feature": "brows",
            "variant": normalize(brows["arch"]),
            "query": f"{brows['arch']} eyebrow shaping"
        })

    # Jawline
    jaw = face_features.get("jaw_chin", {}).get("jaw", "")
    if jaw:
        intents.append({
            "feature": "jawline",
            "variant": normalize(jaw),
            "query": f"{jaw} jawline contour technique"
        })

    # Chin
    chin = face_features.get("jaw_chin", {}).get("chin_shape", "")
    if chin:
        intents.append({
            "feature": "chin",
            "variant": normalize(chin),
            "query": f"{chin} chin contour technique"
        })

    # Cheekbones
    cheeks = face_features.get("cheekbones", {})
    if cheeks.get("prominence"):
        intents.append({
            "feature": "cheekbones",
            "variant": normalize(cheeks["prominence"]),
            "query": f"{cheeks['prominence']} cheekbone makeup technique"
        })

    return intents

# -----------------------------
# Retrieval
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_from_face_features(face_features, top_k=1):
    intents = build_feature_queries(face_features)
    seen = set()
    results_all = []

    for intent in intents:
        key = (intent["feature"], intent["variant"])
        if key in seen:
            continue
        seen.add(key)

        embedding = embedding_model.encode(intent["query"]).tolist()

        # Try feature + variant
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={
                "$and": [
                    {"feature": normalize(intent["feature"])},
                    {"variant": normalize(intent["variant"])}
                ]
            }
        )

        # fallback: feature-only
        if not results["documents"][0]:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where={"feature": normalize(intent["feature"])}
            )

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            steps = []
            if meta.get("steps"):
                try:
                    steps = json.loads(meta["steps"])
                except:
                    steps = []

            results_all.append({
                "feature": meta["feature"],
                "variant": meta["variant"],
                "technique": meta.get("technique", ""),
                "steps": steps,   # ✅ must exist now
                "distance": dist
            })

    return results_all
