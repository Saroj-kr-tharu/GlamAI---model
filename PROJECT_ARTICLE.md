# GlamAI — AI-Powered Face Analysis & Personalized Makeup Recommendation Engine

> *An intelligent system that analyzes facial geometry from a single photo and delivers personalized, step-by-step makeup recommendations using computer vision, anthropometric science, and generative AI.*

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Pipeline Flow](#-pipeline-flow)
4. [Layer 1 — Facial Landmark Extraction](#-layer-1--facial-landmark-extraction)
5. [Layer 2 — Anthropometric Metrics](#-layer-2--anthropometric-metrics)
6. [Layer 3 — Feature Classification](#-layer-3--feature-classification)
7. [RAG-Based Makeup Recommendation Generation](#-rag-based-makeup-recommendation-generation)
8. [Knowledge Base](#-knowledge-base)
9. [API Design](#-api-design)
10. [Deployment Architecture](#-deployment-architecture)
11. [Tech Stack](#-tech-stack)

---

## 🧠 Project Overview

**GlamAI** is a full-stack AI pipeline that transforms a user's selfie into personalized makeup guidance. Rather than relying on generic beauty advice, GlamAI measures the unique geometry of each face — eye shape, nose proportions, lip fullness, jawline structure, and more — then retrieves the most relevant professional makeup techniques from a curated knowledge base, enhanced with AI-generated explanations.

### Key Capabilities

- **478-point facial landmark detection** via Google MediaPipe
- **Anthropometric measurement** of 8 distinct facial regions
- **Rule-based feature classification** grounded in facial morphology science
- **Retrieval-Augmented Generation (RAG)** using vector search (ChromaDB) + LLM (Ollama/Phi3)
- **REST API** served via Flask + Gunicorn, containerized with Docker

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          GlamAI System Architecture                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐     ┌──────────────────────────────────────────────┐     │
│   │  Client   │────▶│              Flask REST API (app.py)         │     │
│   │ (Frontend)│◀────│  POST /analyze  ─  GET /                    │     │
│   └──────────┘     └───────────────────┬──────────────────────────┘     │
│                                        │                                │
│                    ┌───────────────────▼───────────────────┐            │
│                    │        Processing Pipeline             │            │
│                    │                                        │            │
│                    │  ┌────────────────────────────────┐    │            │
│                    │  │  Layer 1: Landmark Extraction   │    │            │
│                    │  │  (MediaPipe FaceLandmarker)     │    │            │
│                    │  └──────────────┬─────────────────┘    │            │
│                    │                 │ 478 (x,y,z) coords   │            │
│                    │  ┌──────────────▼─────────────────┐    │            │
│                    │  │  Layer 2: Metric Calculation    │    │            │
│                    │  │  (Anthropometric Ratios)        │    │            │
│                    │  └──────────────┬─────────────────┘    │            │
│                    │                 │ normalized metrics    │            │
│                    │  ┌──────────────▼─────────────────┐    │            │
│                    │  │  Layer 3: Feature Classification│    │            │
│                    │  │  (Rule-Based Classifier)        │    │            │
│                    │  └──────────────┬─────────────────┘    │            │
│                    │                 │ face_features JSON    │            │
│                    │  ┌──────────────▼─────────────────┐    │            │
│                    │  │  Generation: RAG + LLM          │    │            │
│                    │  │  ChromaDB ◀─── Knowledge Base   │    │            │
│                    │  │  Ollama (Phi3) ◀─── Prompts     │    │            │
│                    │  └────────────────────────────────┘    │            │
│                    └───────────────────────────────────────┘            │
│                                                                         │
│   ┌────────────────┐    ┌──────────────────┐                           │
│   │  ChromaDB       │    │  Ollama (Phi3)    │                           │
│   │  (In-Memory     │    │  (LLM Server)     │                           │
│   │   Vector Store) │    │  Port: 11434      │                           │
│   └────────────────┘    └──────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Flow

The system processes each image through four sequential stages:

```
 ┌─────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
 │  Image   │───▶│   Layer 1    │───▶│   Layer 2     │───▶│   Layer 3     │───▶│  Generation  │
 │  Upload  │    │  Extraction  │    │  Metrics      │    │  Classify     │    │  RAG + LLM   │
 └─────────┘    └─────────────┘    └──────────────┘    └──────────────┘    └──────┬──────┘
                                                                                  │
                                                                                  ▼
                                                                          ┌─────────────┐
                                                                          │   JSON       │
                                                                          │   Response   │
                                                                          │  (Features + │
                                                                          │  Makeup Tips)│
                                                                          └─────────────┘
```

**Data transformations at each stage:**

| Stage | Input | Output | Module |
|-------|-------|--------|--------|
| **Layer 1** | Raw image (PNG/JPG) | 478 landmark coordinates `(x, y, z)` | `layer1_extraction.py` |
| **Layer 2** | Landmark coordinates + image dimensions | ~20 normalized metrics (ratios, angles) | `layer2_metrics.py` |
| **Layer 3** | Normalized metrics | Classified features JSON + human-readable text | `layer3_classify.py` |
| **Generation** | Feature classifications + Knowledge base | Personalized makeup recommendations | `generation.py` + `retrieve.py` |

---

## 👁 Layer 1 — Facial Landmark Extraction

**File:** `layer1_extraction.py`

This layer uses **Google MediaPipe's FaceLandmarker** (Tasks API) to detect **478 facial landmarks** on a single face.

### Process

```
 Raw Image
     │
     ▼
 ┌──────────────┐
 │  Resize to    │  Standardize to 512×512
 │  512 × 512    │
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  BGR → RGB    │  Convert color space for MediaPipe
 │  Conversion   │
 └──────┬───────┘
        │
        ▼
 ┌──────────────────────┐
 │  MediaPipe            │  face_landmarker.task model (~29MB)
 │  FaceLandmarker       │  Float16 precision
 │  Detection            │  Min confidence: 0.5
 └──────┬───────────────┘
        │
        ▼
 478 landmarks as (x_pixel, y_pixel, z_depth)
```

### Key Landmarks Used Downstream

| Landmark Index | Anatomical Location | Used For |
|----------------|---------------------|----------|
| 1 | Nose tip | Nose metrics |
| 10 | Forehead top | Cheekbone height |
| 13, 14 | Upper/lower lip center | Lip metrics |
| 33, 263 | Left/right eye center | Eye spacing |
| 61, 291 | Lip corners | Lip width |
| 98, 327 | Nose wings (left/right) | Nose width |
| 105, 65 | Left brow inner/outer | Brow angle |
| 133, 173 | Left eye corners | Eye width |
| 145, 159 | Left eye top/bottom | Eye height |
| 152 | Chin | Jaw/chin metrics |
| 168 | Nose bridge | Nose length |
| 234, 454 | Jaw/cheek extremes | Jaw width, cheekbones |
| 334, 295 | Right brow inner/outer | Brow angle |
| 362, 386, 374 | Right eye points | Eye width/height |

---

## 📐 Layer 2 — Anthropometric Metrics

**File:** `layer2_metrics.py`

This layer computes **normalized ratios and angles** from the raw landmarks, grounded in established anthropometric science. All distances are normalized by face width or face height to ensure **scale-invariance**.

### Computed Metrics

```
┌─────────────────────────────────────────────────────┐
│              Anthropometric Metrics Map              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  FACE GEOMETRY                                      │
│  ├─ face_width .............. max(x) - min(x)       │
│  ├─ face_height ............. max(y) - min(y)       │
│  └─ face_ratio .............. height / width        │
│                                                     │
│  EYES                                               │
│  ├─ inter_eye_distance ...... |R_eye - L_eye| / W   │
│  ├─ eye_symmetry ............ |L_y - R_y| / H       │
│  ├─ left_eye_width .......... / face_width          │
│  ├─ left_eye_height ......... / face_height         │
│  ├─ right_eye_width ......... / face_width          │
│  └─ right_eye_height ........ / face_height         │
│                                                     │
│  NOSE                                               │
│  ├─ nose_width .............. / face_width          │
│  └─ nose_length ............. / face_height         │
│                                                     │
│  LIPS                                               │
│  ├─ upper_lip_height ........ / face_height         │
│  ├─ lower_lip_height ........ / face_height         │
│  └─ lip_width ............... / face_width          │
│                                                     │
│  EYEBROWS                                           │
│  ├─ left_brow_angle ......... degrees (atan2)       │
│  └─ right_brow_angle ........ degrees (atan2)       │
│                                                     │
│  JAW & CHIN                                         │
│  ├─ jaw_width ............... / face_width          │
│  └─ chin_projection ......... / face_height         │
│                                                     │
│  CHEEKBONES                                         │
│  ├─ cheekbone_prominence .... / face_width          │
│  └─ cheekbone_height ........ / face_height         │
│                                                     │
│  W = face_width    H = face_height                  │
└─────────────────────────────────────────────────────┘
```

---

## 🏷 Layer 3 — Feature Classification

**File:** `layer3_classify.py`

This layer applies **rule-based thresholds** (derived from anthropometric literature) to classify each facial region into descriptive categories.

### Classification Rules

#### Face Symmetry (Eye Alignment)

```
  Eye Alignment:     < 0.015       0.015–0.03       > 0.03
                   ┌──────┐      ┌──────────┐     ┌───────────────────────┐
                   │ High │      │ Moderate │     │ Noticeable Asymmetry │
                   └──────┘      └──────────┘     └───────────────────────┘
```

#### Face Shape (Facial Index = height/width)

```
  Facial Index:    < 0.85      0.85–0.90    0.90–0.95     0.95–1.00     > 1.00
                  ┌──────┐   ┌──────┐     ┌──────┐      ┌──────┐     ┌──────────┐
  Classification: │ Broad │   │ Round│     │ Oval │      │ Long │     │ Very Long│
                  └──────┘   └──────┘     └──────┘      └──────┘     └──────────┘
```

#### Eyes

```
  Eye Ratio (H/W):      > 0.8         0.6–0.8        < 0.6
                       ┌───────┐     ┌─────────┐    ┌────────┐
                       │ Round │     │ Almond  │    │ Hooded │
                       └───────┘     └─────────┘    └────────┘

  Inter-Eye Distance:   < 0.32        0.32–0.36      > 0.36
                       ┌───────────┐ ┌──────────┐  ┌──────────┐
                       │ Close-set │ │ Balanced │  │ Wide-set │
                       └───────────┘ └──────────┘  └──────────┘
```

#### Nose

```
  Width Ratio:    < 0.14       0.14–0.18     > 0.18
                 ┌────────┐   ┌─────────┐   ┌──────┐
                 │ Narrow │   │ Average │   │ Wide │
                 └────────┘   └─────────┘   └──────┘

  Length Ratio:   < 0.28       0.28–0.36     > 0.36
                 ┌───────┐   ┌─────────┐   ┌──────┐
                 │ Short │   │ Average │   │ Long │
                 └───────┘   └─────────┘   └──────┘

  Tip Shape:     short+narrow → Rounded
                 wide         → Soft Curve
                 otherwise    → Defined
```

#### Lips

```
  Fullness (upper+lower):  < 0.05      0.05–0.08     > 0.08
                          ┌──────┐    ┌────────┐    ┌──────┐
                          │ Thin │    │ Medium │    │ Full │
                          └──────┘    └────────┘    └──────┘

  Balance (upper/lower):   > 1.05          0.95–1.05         < 0.95
                       ┌────────────────┐ ┌──────────┐  ┌────────────────┐
                       │ Upper-Dominant │ │ Balanced │  │ Lower-Dominant │
                       └────────────────┘ └──────────┘  └────────────────┘
```

#### Eyebrows

```
  Average Angle:     < 5°          5°–15°          > 15°
                   ┌──────────┐  ┌───────────┐  ┌──────────────┐
                   │ Straight │  │ Soft Arch │  │ Defined Arch │
                   └──────────┘  └───────────┘  └──────────────┘
```

#### Jaw & Chin

```
  Jaw Width Ratio:   < 0.35       0.35–0.45      > 0.45
                    ┌────────┐   ┌──────────┐   ┌──────┐
                    │ Narrow │   │ Balanced │   │ Wide │
                    └────────┘   └──────────┘   └──────┘

  Chin Projection:   < 0.03       0.03–0.05      > 0.05
                    ┌─────────┐  ┌──────────┐   ┌───────────┐
                    │ Pointed │  │ Balanced │   │ Prominent │
                    └─────────┘  └──────────┘   └───────────┘
```

#### Cheekbones

```
  Prominence:      < 0.8         0.8–1.0        > 1.0
                  ┌────────┐   ┌──────────┐   ┌───────────┐
                  │ Subtle │   │ Moderate │   │ Prominent │
                  └────────┘   └──────────┘   └───────────┘
```

### Example Output

```json
{
  "face_shape": { "primary": "oval", "secondary": "round", "ratio": 0.93 },
  "face_symmetry": { "level": "high", "eye_alignment": 0.008 },
  "nose": { "width": "average", "length": "average", "tip": "defined" },
  "eyes": { "shape": "almond", "orientation": "balanced", "spacing": "balanced" },
  "lips": { "fullness": "medium", "balance": "balanced", "contour": "natural" },
  "eyebrows": { "arch": "soft arch", "thickness": "natural" },
  "jaw_chin": { "jaw": "balanced", "chin_shape": "balanced" },
  "cheekbones": { "prominence": "moderate", "height": "high-set" }
}
```

---

## 🤖 RAG-Based Makeup Recommendation Generation

**Files:** `retrieve.py` + `generation.py`

This is the most sophisticated layer — a **Retrieval-Augmented Generation (RAG)** pipeline that combines vector search with LLM reasoning.

### RAG Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                    RAG Recommendation Pipeline                        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐     INDEX PHASE                                 │
│  │  Knowledge Base  │────────────────┐                                │
│  │  (8 JSON files)  │                │                                │
│  └─────────────────┘                ▼                                │
│                            ┌─────────────────┐                        │
│                            │  ChromaDB         │  In-memory vector    │
│                            │  EphemeralClient  │  store (rebuilt      │
│                            │                   │  per request)        │
│                            └────────┬──────────┘                      │
│                                     │                                 │
│  ┌─────────────────┐     RETRIEVAL PHASE                             │
│  │ Face Features    │────────────────┐                                │
│  │ (from Layer 3)   │                │                                │
│  └─────────────────┘                ▼                                │
│                            ┌─────────────────┐                        │
│                            │  Query Builder    │  Builds semantic     │
│                            │  (per feature)    │  queries like:       │
│                            │                   │  "almond eyes        │
│                            │                   │   makeup technique"  │
│                            └────────┬──────────┘                      │
│                                     │                                 │
│                                     ▼                                │
│                            ┌─────────────────┐                        │
│                            │ SentenceTransf.  │  all-MiniLM-L6-v2    │
│                            │ Embedding        │  384-dim embeddings   │
│                            └────────┬──────────┘                      │
│                                     │                                 │
│                                     ▼                                │
│                            ┌─────────────────┐                        │
│                            │ Vector Search     │  Feature + variant   │
│                            │ (ChromaDB)        │  filtered, with      │
│                            │                   │  fallback to         │
│                            │                   │  feature-only        │
│                            └────────┬──────────┘                      │
│                                     │                                 │
│                                     ▼  Retrieved: technique + steps  │
│                                                                       │
│                            GENERATION PHASE                           │
│                            ┌─────────────────┐                        │
│                            │  Prompt Builder   │  Strict instruction: │
│                            │                   │  DO NOT modify steps │
│                            │                   │  Only explain WHY +  │
│                            │                   │  add AWARENESS tips  │
│                            └────────┬──────────┘                      │
│                                     │                                 │
│                                     ▼                                │
│                            ┌─────────────────┐                        │
│                            │  Ollama (Phi3)    │  Local LLM           │
│                            │  Chat API         │  3 retry attempts    │
│                            └────────┬──────────┘                      │
│                                     │                                 │
│                                     ▼                                │
│                            ┌─────────────────┐                        │
│                            │  JSON Extractor   │  Robust parser with  │
│                            │  + Fallbacks      │  code-fence removal, │
│                            │                   │  key normalization,  │
│                            │                   │  & safe fallbacks    │
│                            └─────────────────┘                        │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Per-Feature Recommendation Output

For each classified facial feature, the system produces:

```json
{
  "feature": "eyes",
  "variant": "almond",
  "technique": "crease definition",
  "steps": [
    "Apply light base over lid.",
    "Define crease softly.",
    "Extend liner slightly outward."
  ],
  "why_it_matches": "The crease definition technique enhances almond eyes by...",
  "awareness": "Apply products gently and blend well to maintain a natural look."
}
```

---

## 📚 Knowledge Base

The knowledge base consists of **8 JSON files**, each covering a facial feature with **variant-specific makeup techniques**:

```
knowledge/
├── cheekbones.json     3 variants: subtle, moderate, prominent
├── chin.json           3 variants: pointed, balanced, prominent
├── eyebrows.json       3 variants: straight, soft_arch, defined_arch
├── Eyes.json           6 variants: round, almond, hooded, close_set, wide_set, balanced
├── Face_Shape.json     5 variants: broad, round, oval, long, very_long
├── jawline.json        3 variants: narrow, balanced, wide
├── Lips.json           3 variants: full, defined_cupid_bow, natural
└── Nose.json           3 variants: rounded, soft_curve, defined
                       ─────
                       29 total technique entries
```

### Knowledge Entry Schema

Each entry follows a consistent structure:

```json
{
  "id": "unique_identifier",
  "feature": "feature_name",
  "variant": "variant_name",
  "technique": "technique description",
  "category": "tutorial",
  "tags": ["relevant", "tags"],
  "steps": [
    "Step 1 instruction.",
    "Step 2 instruction.",
    "Step 3 instruction."
  ]
}
```

---

## 🌐 API Design

**File:** `app.py`

The system exposes a clean REST API via **Flask** with **CORS** enabled.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check — returns `{"status": "ok"}` |
| `POST` | `/analyze` | Full face analysis pipeline |

### `POST /analyze`

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` field with a photo (PNG, JPG, JPEG, WEBP, BMP)
- Max size: **10 MB**

**Response (200):**
```json
{
  "success": true,
  "face_features": {
    "face_shape": { "primary": "oval", "secondary": "round", "ratio": 0.93 },
    "eyes": { "shape": "almond", "orientation": "balanced", "spacing": "balanced" },
    "nose": { "width": "average", "length": "average", "tip": "defined" },
    "lips": { "fullness": "medium", "balance": "balanced", "contour": "natural" },
    "eyebrows": { "arch": "soft arch", "thickness": "natural" },
    "jaw_chin": { "jaw": "balanced", "chin_shape": "balanced" },
    "cheekbones": { "prominence": "moderate", "height": "high-set" }
  },
  "human_readable": "Your face shape is oval with subtle round influence.\nYour eyes are almond, balanced, and balanced.\n...",
  "recommendations": [
    {
      "feature": "eyes",
      "variant": "almond",
      "technique": "crease definition",
      "steps": ["Apply light base over lid.", "Define crease softly.", "Extend liner slightly outward."],
      "why_it_matches": "...",
      "awareness": "..."
    }
  ]
}
```

**Error Responses:**

| Code | Scenario |
|------|----------|
| `400` | No image provided / empty filename / unsupported format |
| `422` | No face detected in the image |
| `500` | Internal processing error |

---

## 🐳 Deployment Architecture

**Files:** `Dockerfile` + `docker-compose.yml`

The project uses a **multi-stage Docker build** and **Docker Compose** for orchestration.

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────────────────────────────┐               │
│  │  face-api  (GlamAI Flask API)             │               │
│  │  ├─ Python 3.11-slim                      │               │
│  │  ├─ Gunicorn (2 workers, 2 threads)       │               │
│  │  ├─ Port: 5000                            │               │
│  │  ├─ Memory limit: 2 GB                    │               │
│  │  ├─ Health check: every 30s               │               │
│  │  └─ Depends on: ollama, ollama-pull       │               │
│  └───────────────────┬───────────────────────┘               │
│                      │ HTTP (OLLAMA_HOST)                     │
│                      ▼                                       │
│  ┌───────────────────────────────────────────┐               │
│  │  ollama  (LLM Server)                     │               │
│  │  ├─ ollama/ollama:latest                  │               │
│  │  ├─ Port: 11434                           │               │
│  │  └─ Volume: ollama_data (model cache)     │               │
│  └───────────────────────────────────────────┘               │
│                      ▲                                       │
│                      │ Pulls phi3 model on startup            │
│  ┌───────────────────┴───────────────────────┐               │
│  │  ollama-pull  (Init Container)            │               │
│  │  └─ Runs: sleep 5 && ollama pull phi3     │               │
│  └───────────────────────────────────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Stage Docker Build

```
 Stage 1: Builder                         Stage 2: Runtime
┌─────────────────────────┐              ┌─────────────────────────┐
│ python:3.11-slim         │              │ python:3.11-slim         │
│                          │              │                          │
│ • Install build tools    │              │ • Runtime libs only      │
│ • pip install deps       │─────────────▶│ • Copy /install from     │
│ • Download MediaPipe     │  COPY        │   builder                │
│   face_landmarker.task   │              │ • Copy app code          │
│   (~29MB, verified)      │              │ • Copy verified model    │
│                          │              │ • Gunicorn + tini        │
└─────────────────────────┘              └─────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Web Framework** | Flask 3.1 | REST API server |
| **CORS** | flask-cors 6.0 | Cross-origin support |
| **WSGI Server** | Gunicorn 23.0 | Production HTTP server |
| **Computer Vision** | MediaPipe 0.10.32 | 478-point face landmark detection |
| **Image Processing** | OpenCV (headless) 4.13 | Image loading, resizing, color conversion |
| **Vector Database** | ChromaDB 1.5 | In-memory semantic search for knowledge retrieval |
| **Embeddings** | sentence-transformers 5.2 | `all-MiniLM-L6-v2` for query/document embeddings |
| **LLM** | Ollama 0.5 + Phi3 | Local LLM for generating explanations |
| **Numerical** | NumPy 2.2 | Array operations |
| **Config** | python-dotenv 1.1 | Environment variable management |
| **Containerization** | Docker + Docker Compose | Multi-stage build and service orchestration |
| **Process Manager** | tini | PID-1 signal handling in containers |

---

## 📊 Summary: End-to-End Data Flow

```
  📸 User Photo
       │
       ▼
  ┌─────────────┐
  │  MediaPipe   │──▶ 478 landmarks (x, y, z)
  └─────────────┘
       │
       ▼
  ┌─────────────┐
  │  Metrics     │──▶ ~20 normalized ratios & angles
  └─────────────┘
       │
       ▼
  ┌─────────────┐
  │  Classifier  │──▶ 8 feature categories with labels
  └─────────────┘         (face_shape, eyes, nose, lips,
       │                   eyebrows, jaw, chin, cheekbones)
       ▼
  ┌─────────────┐    ┌──────────────┐
  │  ChromaDB   │◀───│ 29 Knowledge │
  │  Retrieval  │    │ Entries      │
  └──────┬──────┘    └──────────────┘
         │
         ▼
  ┌─────────────┐
  │  Phi3 LLM   │──▶ Per-feature recommendations with:
  └─────────────┘     • Technique name
                      • Step-by-step instructions (from KB)
                      • Why it matches (LLM-generated)
                      • Awareness tips (LLM-generated)
       │
       ▼
  📋 JSON Response to Client
```

---

*GlamAI — Where computer vision meets beauty science. Every face tells a story; GlamAI helps you enhance it.*
