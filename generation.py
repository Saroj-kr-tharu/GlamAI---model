import json
import re
import ollama
from retrieve import retrieve_from_face_features, load_face_features, index_knowledge


# ---------------------------
# JSON extractor (SAFE)
# ---------------------------
def extract_json(raw_text):
    """
    Robust JSON extractor for LLM outputs.
    Returns a dict with:
        - why_it_matches: always string
        - awareness: always string
    """
    if not raw_text:
        return {"why_it_matches": "", "awareness": ""}

    # Remove code fences and unwanted chars
    text = re.sub(r"```json|```", "", raw_text, flags=re.IGNORECASE).strip()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"(\w+):", r'"\1":', text)

    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = json.loads(text.replace("'", '"'))
        except Exception:
            print(f"Warning: JSON parse failed. Returning empty.\nRaw output: {raw_text}")
            return {"why_it_matches": "", "awareness": ""}

    # Flatten keys
    why_keys = ["why_it_matches", "whyItMatches", "whyItMatchesReasoning"]
    awareness_keys = ["awareness", "Awareness", "warning", "tip"]

    combined_why = []
    combined_awareness = []

    def extract_fields(obj):
        w = ""
        a = ""
        for k in why_keys:
            if k in obj and obj[k]:
                w = str(obj[k])
                break
        for k in awareness_keys:
            if k in obj and obj[k]:
                # Flatten list/dict to string
                val = obj[k]
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=False)
                a = str(val)
                break
        return w.strip(), a.strip()

    if isinstance(parsed, dict):
        w, a = extract_fields(parsed)
        if w: combined_why.append(w)
        if a: combined_awareness.append(a)

        if "steps" in parsed and isinstance(parsed["steps"], list):
            for step in parsed["steps"]:
                if isinstance(step, dict):
                    w_step, a_step = extract_fields(step)
                    if w_step: combined_why.append(w_step)
                    if a_step: combined_awareness.append(a_step)

    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                w, a = extract_fields(item)
                if w: combined_why.append(w)
                if a: combined_awareness.append(a)

    return {
        "why_it_matches": " ".join(combined_why).strip() or "LLM failed to generate explanation",
        "awareness": " ".join(combined_awareness).strip() or "No awareness tip"
    }

# ---------------------------
# Prompt builder
# ---------------------------
def build_prompt(feature_data):
    feature = feature_data["feature"]
    variant = feature_data["variant"]
    technique = feature_data.get("technique") or "No specific technique provided"
    steps = feature_data.get("steps", [])

    # Join steps for display in prompt
    steps_text = json.dumps(steps, ensure_ascii=False, indent=2)

    prompt = f"""
You are a professional makeup educator.

STRICT RULES:
- You MUST NOT create, modify, reorder, or rephrase steps
- Steps are PROVIDED and MUST remain EXACTLY the same
- You are ONLY allowed to:
  1. Explain why this technique suits the feature
  2. Add awareness or caution notes

Feature: {feature}
Variant: {variant}
Technique: {technique}

Provided steps (DO NOT CHANGE):
{steps_text}

Return ONLY valid JSON in this EXACT structure:
{{
  "why_it_matches": "clear, concise explanation",
  "awareness": "simple precaution or tip"
}}
"""
    return prompt


# ---------------------------
# Generate recommendation
# ---------------------------
def generate_recommendation(feature_data, model_name="phi3", max_retries=3):
    """
    Generates why_it_matches and awareness for a feature.
    Ensures frontend-safe strings and provides fallbacks if LLM fails.
    """
    feature_name = feature_data.get("feature", "unknown")
    if not feature_data.get("steps"):
        return {"why_it_matches": "No steps available", "awareness": ""}

    prompt = build_prompt(feature_data)

    parsed = {}
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response["message"]["content"]
            parsed = extract_json(content)

            if parsed.get("why_it_matches") or parsed.get("awareness"):
                break  # Successfully generated

        except Exception as e:
            print(f"Attempt {attempt+1} failed for {feature_name}: {e}")

    # --- FALLBACKS if LLM failed ---
    if not parsed.get("why_it_matches") or "LLM failed" in parsed.get("why_it_matches"):
        feature = feature_data.get("feature", "feature")
        variant = feature_data.get("variant", "")
        technique = feature_data.get("technique", "")
        parsed["why_it_matches"] = (
            f"This technique ({technique}) suits the {variant} {feature} "
            "by enhancing natural features and maintaining balance."
            )

    if not parsed.get("awareness") or "No awareness tip" in parsed.get("awareness"):
        parsed["awareness"] = (
        "Apply products gently and blend well to maintain a natural look."
        )
    # Flatten any lists/dicts in awareness for frontend safety
    if isinstance(parsed["awareness"], (list, dict)):
        parsed["awareness"] = json.dumps(parsed["awareness"], ensure_ascii=False)

    return parsed

def run_generation(
    face_features_path="face_features.json",
    knowledge_path="./knowledge",
    output_path="final_makeup_recommendations.json",
    model_name="phi3"
):
    print("Indexing knowledge base...")
    index_knowledge(knowledge_path)

    print("Retrieving techniques...")
    face_features = load_face_features(face_features_path)
    retrieval_results = retrieve_from_face_features(face_features, top_k=1)

    final_recommendations = []

    for i, feature_data in enumerate(retrieval_results, 1):
        feature_name = feature_data.get("feature", "unknown")
        print(f"Processing {i}/{len(retrieval_results)} -> {feature_name}")

        generated = generate_recommendation(
            feature_data,
            model_name=model_name
        )

        final_recommendations.append({
            "feature": feature_name,
            "variant": feature_data.get("variant", ""),
            "technique": feature_data.get("technique", ""),
            "steps": feature_data.get("steps", []),
            "why_it_matches": generated.get("why_it_matches", ""),
            "awareness": generated.get("awareness", "")
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_recommendations, f, ensure_ascii=False, indent=4)

    print("\nGeneration complete!")
    return final_recommendations

# ---------------------------
# Main pipeline
# ---------------------------
if __name__ == "__main__":
  if __name__ == "__main__":
    run_generation()

