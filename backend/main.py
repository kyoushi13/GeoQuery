# # backend/main.py
# import io
# import torch
# import base64
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
# import gc

# def cleanup():
#     gc.collect()
#     if torch.backends.mps.is_available():
#         torch.mps.empty_cache()

# from detect import detect, draw_boxes, CLASSES
# from caption_vqa import caption, answer

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MAX_RES = 1024
# sessions = {}  # simple in-memory store: {session_id: {"image": PIL.Image, "history": [...]}}

# def img_to_b64(img: Image.Image) -> str:
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode()

# @app.post("/upload")
# async def upload(file: UploadFile = File(...), session_id: str = Form("default")):
#     if file.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Unsupported file type. Use .jpg or .png only.")

#     contents = await file.read()
#     try:
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Could not read image file.")

#     if image.width > MAX_RES or image.height > MAX_RES:
#         raise HTTPException(status_code=400, detail=f"Image exceeds max resolution {MAX_RES}x{MAX_RES}.")

#     cap = caption(image)
#     dets = detect(image)
#     boxed_img = draw_boxes(image, dets)

#     sessions[session_id] = {"image": image, "history": []}
#     sessions[session_id]["history"].append({"role": "system", "content": f"Image uploaded. Caption: {cap}"})
#     cleanup()
#     return {
#         "caption": cap,
#         "detections": dets,
#         "annotated_image": img_to_b64(boxed_img)
#     }

# DETECTION_KEYWORDS = ["mark", "detect", "highlight", "locate", "find all", "show all"]

# @app.post("/query")
# async def query(question: str = Form(...), session_id: str = Form("default")):
#     if session_id not in sessions:
#         raise HTTPException(status_code=400, detail="No image uploaded for this session.")

#     image = sessions[session_id]["image"]
#     q_lower = question.lower()

#     # out-of-scope check
#     known_terms = CLASSES + ["object", "objects", "image", "picture", "them", "those", "it"]
#     # simple heuristic: skip strict validation, let model attempt; only hard-block obvious mismatches
#     # (kept lightweight per hackathon time constraints)

#     is_detection_query = any(kw in q_lower for kw in DETECTION_KEYWORDS)

#     if is_detection_query:
#         dets = detect(image)
#         boxed_img = draw_boxes(image, dets)
#         response = {
#             "type": "detection",
#             "detections": dets,
#             "annotated_image": img_to_b64(boxed_img),
#             "text": f"Found {len(dets)} object(s)." if dets else "No matching objects found."
#         }
#     else:
#         # numeric questions: count via detection instead of trusting VQA math
#         if any(w in q_lower for w in ["how many", "count", "number of"]):
#             dets = detect(image)
#             target_class = None
#             for c in CLASSES:
#                 if c.split()[0] in q_lower or c in q_lower:
#                     target_class = c
#                     break
#             if target_class:
#                 count = sum(1 for d in dets if d["class"] == target_class)
#                 text = f"There are {count} {target_class}(s) in the image."
#             else:
#                 text = answer(image, question)

#             response = {"type": "vqa_numeric", "text": text}

#         elif any(w in q_lower for w in ["is there", "are there", "does the image"]):
#             dets = detect(image)
#             for c in CLASSES:  # use original labels
#                 if c in q_lower:
#                     present = any(d["class"] == c for d in dets)
#                     text = "Yes." if present else "No."
#                     break
#             else:
#                 text = answer(image, question)  # fallback to BLIP

#             response = {
#             "type": "vqa_binary",
#             "text": text
# }

#         else:
#             ans = answer(image, question)
#             response = {"type": "vqa", "text": ans}

#     sessions[session_id]["history"].append({"role": "user", "content": question})
#     MAX_SESSIONS = 5
#     if len(sessions) > MAX_SESSIONS:
#         sessions.pop(next(iter(sessions)))
#     sessions[session_id]["history"].append({"role": "assistant", "content": response["text"]})
#     cleanup()
#     return response

# @app.get("/history")
# async def history(session_id: str = "default"):
#     if session_id not in sessions:
#         return {"history": []}
#     return {"history": sessions[session_id]["history"]}
# backend/main.py
import io
import base64
import gc
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from detect import detect, draw_boxes, CLASSES
from caption_vqa import caption, answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_RES = 1024
MAX_SESSIONS = 5
sessions = {}  # {session_id: {"image": PIL.Image, "detections": [...], "history": [...]}}

def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

UNSUPPORTED_MSG = (
    "GeoQuery supports only:\n"
    "- building\n- road\n- vehicle\n- vegetation\n- water body\n- open ground"
)

# common out-of-scope object words to catch (extend as needed)
UNSUPPORTED_TERMS = [
    "airplane", "plane", "aircraft", "person", "people", "animal", "dog", "cat",
    "bicycle", "boat", "ship", "train", "traffic light", "sign", "bridge"
]

HIGHLIGHT_KEYWORDS = ["highlight", "mark", "show", "detect", "locate", "find all"]

@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str = Form("default")):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .jpg or .png only.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file. File may be corrupted.")

    if image.width > MAX_RES or image.height > MAX_RES:
        raise HTTPException(status_code=400, detail=f"Image exceeds max resolution {MAX_RES}x{MAX_RES}.")

    cap = caption(image)
    dets, resized = detect(image)
    boxed_img = draw_boxes(resized, dets)

    if len(sessions) >= MAX_SESSIONS:
        sessions.pop(next(iter(sessions)))

    sessions[session_id] = {
        "image": resized,          # store resized image, keeps coords consistent for all future ops
        "detections": dets,        # cached — reused for numeric/binary/highlight, no re-inference
        "history": []
    }
    sessions[session_id]["history"].append({"role": "system", "content": f"Image uploaded. Caption: {cap}"})

    cleanup()
    return {
        "caption": cap,
        "detections": [
            {"class": d["class"], "confidence": d["confidence"], "size": d["size"], "color": d["color"]}
            for d in dets
        ],
        "annotated_image": img_to_b64(boxed_img)
    }

def find_matched_classes(q_lower):
    """Return list of supported classes mentioned in query."""
    matches = []
    for c in CLASSES:
        # handle singular/plural + partial word (e.g. "vehicles" matches "vehicle")
        if c in q_lower or c.split()[0] in q_lower:
            matches.append(c)
    return matches

def has_unsupported_term(q_lower):
    return any(term in q_lower for term in UNSUPPORTED_TERMS)

@app.post("/query")
async def query(question: str = Form(...), session_id: str = Form("default")):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session. Please upload an image first.")

    session = sessions[session_id]
    image = session["image"]
    dets = session["detections"]  # cached, no re-inference
    q_lower = question.lower()

    matched_classes = find_matched_classes(q_lower)

    # --- Unsupported object check (before anything else, no hallucination) ---
    if has_unsupported_term(q_lower) and not matched_classes:
        response = {"type": "unsupported", "text": UNSUPPORTED_MSG}
        session["history"].append({"role": "user", "content": question})
        session["history"].append({"role": "assistant", "content": response["text"]})
        return response

    is_highlight_query = any(kw in q_lower for kw in HIGHLIGHT_KEYWORDS)

    # --- Highlight/detection query: filter by class, return filtered annotated image ---
    if is_highlight_query:
        if matched_classes:
            filtered = [d for d in dets if d["class"] in matched_classes]
        else:
            filtered = dets  # no specific class named -> highlight everything

        boxed_img = draw_boxes(image, filtered)
        text = (
            f"Highlighted {len(filtered)} object(s)"
            + (f" of class '{', '.join(matched_classes)}'." if matched_classes else ".")
        )
        response = {
            "type": "detection",
            "detections": [
                {"class": d["class"], "confidence": d["confidence"], "size": d["size"], "color": d["color"]}
                for d in filtered
            ],
            "annotated_image": img_to_b64(boxed_img),
            "text": text
        }

    # --- Numeric query: count from cached detections, never call VQA ---
    elif any(w in q_lower for w in ["how many", "count", "number of", "how much"]):
        if matched_classes:
            target = matched_classes[0]
            count = sum(1 for d in dets if d["class"] == target)
            text = f"There are {count} {target}(s) in the image." if target != "open ground" and target != "vegetation" else f"Detected {count} region(s) of {target}."
        else:
            text = UNSUPPORTED_MSG
        response = {"type": "vqa_numeric", "text": text}

    # --- Binary query: presence check from cached detections, never call VQA ---
    elif any(w in q_lower for w in ["is there", "are there", "does the image", "any "]):
        if matched_classes:
            target = matched_classes[0]
            present = any(d["class"] == target for d in dets)
            text = "Yes." if present else "No."
        else:
            text = UNSUPPORTED_MSG
        response = {"type": "vqa_binary", "text": text}

    # --- Attribute query: only these go to BLIP VQA ---
    elif any(kw in q_lower for kw in ["what colour", "what color", "which building", "describe", "what is"]):
        ans = answer(image, question)
        response = {"type": "vqa_attribute", "text": ans}

    # --- fallback: unrecognized phrasing, try VQA anyway ---
    else:
        ans = answer(image, question)
        response = {"type": "vqa", "text": ans}

    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": response["text"]})
    cleanup()
    return response

@app.get("/history")
async def history(session_id: str = "default"):
    if session_id not in sessions:
        return {"history": []}
    return {"history": sessions[session_id]["history"]}