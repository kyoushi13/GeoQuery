# # # backend/main.py
# # import io
# # import torch
# # import base64
# # from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse
# # from PIL import Image
# # import gc

# # def cleanup():
# #     gc.collect()
# #     if torch.backends.mps.is_available():
# #         torch.mps.empty_cache()

# # from detect import detect, draw_boxes, CLASSES
# # from caption_vqa import caption, answer

# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # MAX_RES = 1024
# # sessions = {}  # simple in-memory store: {session_id: {"image": PIL.Image, "history": [...]}}

# # def img_to_b64(img: Image.Image) -> str:
# #     buf = io.BytesIO()
# #     img.save(buf, format="PNG")
# #     return base64.b64encode(buf.getvalue()).decode()

# # @app.post("/upload")
# # async def upload(file: UploadFile = File(...), session_id: str = Form("default")):
# #     if file.content_type not in ["image/jpeg", "image/png"]:
# #         raise HTTPException(status_code=400, detail="Unsupported file type. Use .jpg or .png only.")

# #     contents = await file.read()
# #     try:
# #         image = Image.open(io.BytesIO(contents)).convert("RGB")
# #     except Exception:
# #         raise HTTPException(status_code=400, detail="Could not read image file.")

# #     if image.width > MAX_RES or image.height > MAX_RES:
# #         raise HTTPException(status_code=400, detail=f"Image exceeds max resolution {MAX_RES}x{MAX_RES}.")

# #     cap = caption(image)
# #     dets = detect(image)
# #     boxed_img = draw_boxes(image, dets)

# #     sessions[session_id] = {"image": image, "history": []}
# #     sessions[session_id]["history"].append({"role": "system", "content": f"Image uploaded. Caption: {cap}"})
# #     cleanup()
# #     return {
# #         "caption": cap,
# #         "detections": dets,
# #         "annotated_image": img_to_b64(boxed_img)
# #     }

# # DETECTION_KEYWORDS = ["mark", "detect", "highlight", "locate", "find all", "show all"]

# # @app.post("/query")
# # async def query(question: str = Form(...), session_id: str = Form("default")):
# #     if session_id not in sessions:
# #         raise HTTPException(status_code=400, detail="No image uploaded for this session.")

# #     image = sessions[session_id]["image"]
# #     q_lower = question.lower()

# #     # out-of-scope check
# #     known_terms = CLASSES + ["object", "objects", "image", "picture", "them", "those", "it"]
# #     # simple heuristic: skip strict validation, let model attempt; only hard-block obvious mismatches
# #     # (kept lightweight per hackathon time constraints)

# #     is_detection_query = any(kw in q_lower for kw in DETECTION_KEYWORDS)

# #     if is_detection_query:
# #         dets = detect(image)
# #         boxed_img = draw_boxes(image, dets)
# #         response = {
# #             "type": "detection",
# #             "detections": dets,
# #             "annotated_image": img_to_b64(boxed_img),
# #             "text": f"Found {len(dets)} object(s)." if dets else "No matching objects found."
# #         }
# #     else:
# #         # numeric questions: count via detection instead of trusting VQA math
# #         if any(w in q_lower for w in ["how many", "count", "number of"]):
# #             dets = detect(image)
# #             target_class = None
# #             for c in CLASSES:
# #                 if c.split()[0] in q_lower or c in q_lower:
# #                     target_class = c
# #                     break
# #             if target_class:
# #                 count = sum(1 for d in dets if d["class"] == target_class)
# #                 text = f"There are {count} {target_class}(s) in the image."
# #             else:
# #                 text = answer(image, question)

# #             response = {"type": "vqa_numeric", "text": text}

# #         elif any(w in q_lower for w in ["is there", "are there", "does the image"]):
# #             dets = detect(image)
# #             for c in CLASSES:  # use original labels
# #                 if c in q_lower:
# #                     present = any(d["class"] == c for d in dets)
# #                     text = "Yes." if present else "No."
# #                     break
# #             else:
# #                 text = answer(image, question)  # fallback to BLIP

# #             response = {
# #             "type": "vqa_binary",
# #             "text": text
# # }

# #         else:
# #             ans = answer(image, question)
# #             response = {"type": "vqa", "text": ans}

# #     sessions[session_id]["history"].append({"role": "user", "content": question})
# #     MAX_SESSIONS = 5
# #     if len(sessions) > MAX_SESSIONS:
# #         sessions.pop(next(iter(sessions)))
# #     sessions[session_id]["history"].append({"role": "assistant", "content": response["text"]})
# #     cleanup()
# #     return response

# # @app.get("/history")
# # async def history(session_id: str = "default"):
# #     if session_id not in sessions:
# #         return {"history": []}
# #     return {"history": sessions[session_id]["history"]}
# # backend/main.py
# import io
# import base64
# import gc
# import torch
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image

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
# MAX_SESSIONS = 5
# sessions = {}  # {session_id: {"image": PIL.Image, "detections": [...], "history": [...]}}

# def cleanup():
#     gc.collect()
#     if torch.backends.mps.is_available():
#         torch.mps.empty_cache()

# def img_to_b64(img: Image.Image) -> str:
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode()

# UNSUPPORTED_MSG = (
#     "GeoQuery supports only:\n"
#     "- building\n- road\n- vehicle\n- vegetation\n- water body\n- open ground"
# )

# # common out-of-scope object words to catch (extend as needed)
# UNSUPPORTED_TERMS = [
#     "airplane", "plane", "aircraft", "person", "people", "animal", "dog", "cat",
#     "bicycle", "boat", "ship", "train", "traffic light", "sign", "bridge"
# ]

# HIGHLIGHT_KEYWORDS = ["highlight", "mark", "show", "detect", "locate", "find all"]

# @app.post("/upload")
# async def upload(file: UploadFile = File(...), session_id: str = Form("default")):
#     if file.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Unsupported file type. Use .jpg or .png only.")

#     contents = await file.read()
#     try:
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Could not read image file. File may be corrupted.")

#     if image.width > MAX_RES or image.height > MAX_RES:
#         raise HTTPException(status_code=400, detail=f"Image exceeds max resolution {MAX_RES}x{MAX_RES}.")

#     cap = caption(image)
#     dets, resized = detect(image)
#     boxed_img = draw_boxes(resized, dets)

#     if len(sessions) >= MAX_SESSIONS:
#         sessions.pop(next(iter(sessions)))

#     sessions[session_id] = {
#         "image": resized,          # store resized image, keeps coords consistent for all future ops
#         "detections": dets,        # cached — reused for numeric/binary/highlight, no re-inference
#         "history": []
#     }
#     sessions[session_id]["history"].append({"role": "system", "content": f"Image uploaded. Caption: {cap}"})

#     cleanup()
#     return {
#         "caption": cap,
#         "detections": [
#             {"class": d["class"], "confidence": d["confidence"], "size": d["size"], "color": d["color"]}
#             for d in dets
#         ],
#         "annotated_image": img_to_b64(boxed_img)
#     }

# def find_matched_classes(q_lower):
#     """Return list of supported classes mentioned in query."""
#     matches = []
#     for c in CLASSES:
#         # handle singular/plural + partial word (e.g. "vehicles" matches "vehicle")
#         if c in q_lower or c.split()[0] in q_lower:
#             matches.append(c)
#     return matches

# def has_unsupported_term(q_lower):
#     return any(term in q_lower for term in UNSUPPORTED_TERMS)

# @app.post("/query")
# async def query(question: str = Form(...), session_id: str = Form("default")):
#     question = question.strip()
#     if not question:
#         raise HTTPException(status_code=400, detail="Question cannot be empty.")

#     if session_id not in sessions:
#         raise HTTPException(status_code=400, detail="Invalid session. Please upload an image first.")

#     session = sessions[session_id]
#     image = session["image"]
#     dets = session["detections"]  # cached, no re-inference
#     q_lower = question.lower()

#     matched_classes = find_matched_classes(q_lower)

#     # --- Unsupported object check (before anything else, no hallucination) ---
#     if has_unsupported_term(q_lower) and not matched_classes:
#         response = {"type": "unsupported", "text": UNSUPPORTED_MSG}
#         session["history"].append({"role": "user", "content": question})
#         session["history"].append({"role": "assistant", "content": response["text"]})
#         return response

#     is_highlight_query = any(kw in q_lower for kw in HIGHLIGHT_KEYWORDS)

#     # --- Highlight/detection query: filter by class, return filtered annotated image ---
#     if is_highlight_query:
#         if matched_classes:
#             filtered = [d for d in dets if d["class"] in matched_classes]
#         else:
#             filtered = dets  # no specific class named -> highlight everything

#         boxed_img = draw_boxes(image, filtered)
#         text = (
#             f"Highlighted {len(filtered)} object(s)"
#             + (f" of class '{', '.join(matched_classes)}'." if matched_classes else ".")
#         )
#         response = {
#             "type": "detection",
#             "detections": [
#                 {"class": d["class"], "confidence": d["confidence"], "size": d["size"], "color": d["color"]}
#                 for d in filtered
#             ],
#             "annotated_image": img_to_b64(boxed_img),
#             "text": text
#         }

#     # --- Numeric query: count from cached detections, never call VQA ---
#     elif any(w in q_lower for w in ["how many", "count", "number of", "how much"]):
#         if matched_classes:
#             target = matched_classes[0]
#             count = sum(1 for d in dets if d["class"] == target)
#             text = f"There are {count} {target}(s) in the image." if target != "open ground" and target != "vegetation" else f"Detected {count} region(s) of {target}."
#         else:
#             text = UNSUPPORTED_MSG
#         response = {"type": "vqa_numeric", "text": text}

#     # --- Binary query: presence check from cached detections, never call VQA ---
#     elif any(w in q_lower for w in ["is there", "are there", "does the image", "any "]):
#         if matched_classes:
#             target = matched_classes[0]
#             present = any(d["class"] == target for d in dets)
#             text = "Yes." if present else "No."
#         else:
#             text = UNSUPPORTED_MSG
#         response = {"type": "vqa_binary", "text": text}

#     # --- Attribute query: only these go to BLIP VQA ---
#     elif any(kw in q_lower for kw in ["what colour", "what color", "which building", "describe", "what is"]):
#         ans = answer(image, question)
#         response = {"type": "vqa_attribute", "text": ans}

#     # --- fallback: unrecognized phrasing, try VQA anyway ---
#     else:
#         ans = answer(image, question)
#         response = {"type": "vqa", "text": ans}

#     session["history"].append({"role": "user", "content": question})
#     session["history"].append({"role": "assistant", "content": response["text"]})
#     cleanup()
#     return response

# @app.get("/history")
# async def history(session_id: str = "default"):
#     if session_id not in sessions:
#         return {"history": []}
#     return {"history": sessions[session_id]["history"]}

# # # backend/main.py
# # import io
# # import torch
# # import base64
# # from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse
# # from PIL import Image
# # import gc

# # def cleanup():
# #     gc.collect()
# #     if torch.backends.mps.is_available():
# #         torch.mps.empty_cache()

# # from detect import detect, draw_boxes, CLASSES
# # from caption_vqa import caption, answer

# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # MAX_RES = 1024
# # sessions = {}  # simple in-memory store: {session_id: {"image": PIL.Image, "history": [...]}}

# # def img_to_b64(img: Image.Image) -> str:
# #     buf = io.BytesIO()
# #     img.save(buf, format="PNG")
# #     return base64.b64encode(buf.getvalue()).decode()

# # @app.post("/upload")
# # async def upload(file: UploadFile = File(...), session_id: str = Form("default")):
# #     if file.content_type not in ["image/jpeg", "image/png"]:
# #         raise HTTPException(status_code=400, detail="Unsupported file type. Use .jpg or .png only.")

# #     contents = await file.read()
# #     try:
# #         image = Image.open(io.BytesIO(contents)).convert("RGB")
# #     except Exception:
# #         raise HTTPException(status_code=400, detail="Could not read image file.")

# #     if image.width > MAX_RES or image.height > MAX_RES:
# #         raise HTTPException(status_code=400, detail=f"Image exceeds max resolution {MAX_RES}x{MAX_RES}.")

# #     cap = caption(image)
# #     dets = detect(image)
# #     boxed_img = draw_boxes(image, dets)

# #     sessions[session_id] = {"image": image, "history": []}
# #     sessions[session_id]["history"].append({"role": "system", "content": f"Image uploaded. Caption: {cap}"})
# #     cleanup()
# #     return {
# #         "caption": cap,
# #         "detections": dets,
# #         "annotated_image": img_to_b64(boxed_img)
# #     }

# # DETECTION_KEYWORDS = ["mark", "detect", "highlight", "locate", "find all", "show all"]

# # @app.post("/query")
# # async def query(question: str = Form(...), session_id: str = Form("default")):
# #     if session_id not in sessions:
# #         raise HTTPException(status_code=400, detail="No image uploaded for this session.")

# #     image = sessions[session_id]["image"]
# #     q_lower = question.lower()

# #     # out-of-scope check
# #     known_terms = CLASSES + ["object", "objects", "image", "picture", "them", "those", "it"]
# #     # simple heuristic: skip strict validation, let model attempt; only hard-block obvious mismatches
# #     # (kept lightweight per hackathon time constraints)

# #     is_detection_query = any(kw in q_lower for kw in DETECTION_KEYWORDS)

# #     if is_detection_query:
# #         dets = detect(image)
# #         boxed_img = draw_boxes(image, dets)
# #         response = {
# #             "type": "detection",
# #             "detections": dets,
# #             "annotated_image": img_to_b64(boxed_img),
# #             "text": f"Found {len(dets)} object(s)." if dets else "No matching objects found."
# #         }
# #     else:
# #         # numeric questions: count via detection instead of trusting VQA math
# #         if any(w in q_lower for w in ["how many", "count", "number of"]):
# #             dets = detect(image)
# #             target_class = None
# #             for c in CLASSES:
# #                 if c.split()[0] in q_lower or c in q_lower:
# #                     target_class = c
# #                     break
# #             if target_class:
# #                 count = sum(1 for d in dets if d["class"] == target_class)
# #                 text = f"There are {count} {target_class}(s) in the image."
# #             else:
# #                 text = answer(image, question)

# #             response = {"type": "vqa_numeric", "text": text}

# #         elif any(w in q_lower for w in ["is there", "are there", "does the image"]):
# #             dets = detect(image)
# #             for c in CLASSES:  # use original labels
# #                 if c in q_lower:
# #                     present = any(d["class"] == c for d in dets)
# #                     text = "Yes." if present else "No."
# #                     break
# #             else:
# #                 text = answer(image, question)  # fallback to BLIP

# #             response = {
# #             "type": "vqa_binary",
# #             "text": text
# # }

# #         else:
# #             ans = answer(image, question)
# #             response = {"type": "vqa", "text": ans}

# #     sessions[session_id]["history"].append({"role": "user", "content": question})
# #     MAX_SESSIONS = 5
# #     if len(sessions) > MAX_SESSIONS:
# #         sessions.pop(next(iter(sessions)))
# #     sessions[session_id]["history"].append({"role": "assistant", "content": response["text"]})
# #     cleanup()
# #     return response

# # @app.get("/history")
# # async def history(session_id: str = "default"):
# #     if session_id not in sessions:
# #         return {"history": []}
# #     return {"history": sessions[session_id]["history"]}
# # backend/main.py
# import io
# import base64
# import gc
# import torch
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image

# from detect import detect, draw_boxes, CLASSES
# from caption_vqa import caption, answer
# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas as pdf_canvas
# from reportlab.lib.utils import ImageReader
# from fastapi.responses import StreamingResponse

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MAX_RES = 1024
# MAX_SESSIONS = 5
# sessions = {}  # {session_id: {"image": PIL.Image, "detections": [...], "history": [...]}}

# def cleanup():
#     gc.collect()
#     if torch.backends.mps.is_available():
#         torch.mps.empty_cache()

# def img_to_b64(img: Image.Image) -> str:
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode()

# UNSUPPORTED_MSG = (
#     "GeoQuery supports only:\n"
#     "- building\n- road\n- vehicle\n- vegetation\n- water body\n- open ground"
# )

# # common out-of-scope object words to catch (extend as needed)
# UNSUPPORTED_TERMS = [
#     "airplane", "plane", "aircraft", "person", "people", "animal", "dog", "cat",
#     "bicycle", "boat", "ship", "train", "traffic light", "sign", "bridge"
# ]

# HIGHLIGHT_KEYWORDS = ["highlight", "mark", "show", "detect", "locate", "find all"]

# @app.post("/upload")
# async def upload(file: UploadFile = File(...), session_id: str = Form("default")):
#     if file.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Unsupported file type. Use .jpg or .png only.")

#     contents = await file.read()
#     try:
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Could not read image file. File may be corrupted.")

#     if image.width > MAX_RES or image.height > MAX_RES:
#         raise HTTPException(status_code=400, detail=f"Image exceeds max resolution {MAX_RES}x{MAX_RES}.")

#     cap = caption(image)
#     dets, resized = detect(image)
#     boxed_img = draw_boxes(resized, dets)

#     if len(sessions) >= MAX_SESSIONS:
#         sessions.pop(next(iter(sessions)))

#     # sessions[session_id] = {
#     #     "image": resized,          # store resized image, keeps coords consistent for all future ops
#     #     "detections": dets,        # cached — reused for numeric/binary/highlight, no re-inference
#     #     "history": []
#     # }
#     sessions[session_id] = {
#         "image": resized,
#         "detections": dets,
#         "last_referenced_class": None,   # NEW
#         "history": []
#     }
#     sessions[session_id]["history"].append({"role": "system", "content": f"Image uploaded. Caption: {cap}"})

#     cleanup()
#     return {
#         "caption": cap,
#         "detections_full": dets,  # includes id, bbox, class, confidence, size, color — for click overlay
#         "detections": [
#             {"class": d["class"], "confidence": d["confidence"], "size": d["size"], "color": d["color"]}
#             for d in dets
#         ],
#         "annotated_image": img_to_b64(boxed_img)
        
#     }

# def find_matched_classes(q_lower):
#     """Return list of supported classes mentioned in query."""
#     matches = []
#     for c in CLASSES:
#         # handle singular/plural + partial word (e.g. "vehicles" matches "vehicle")
#         if c in q_lower or c.split()[0] in q_lower:
#             matches.append(c)
#     return matches

# def has_unsupported_term(q_lower):
#     return any(term in q_lower for term in UNSUPPORTED_TERMS)

# @app.post("/query")
# async def query(question: str = Form(...), session_id: str = Form("default")):
#     question = question.strip()
#     if not question:
#         raise HTTPException(status_code=400, detail="Question cannot be empty.")

#     if session_id not in sessions:
#         raise HTTPException(status_code=400, detail="Invalid session. Please upload an image first.")

#     session = sessions[session_id]
#     image = session["image"]
#     dets = session["detections"]  # cached, no re-inference
#     q_lower = question.lower()

#     matched_classes = find_matched_classes(q_lower)
#     # resolve pronouns ("those", "them", "it") to last referenced class
#     PRONOUN_TERMS = ["those", "them", "these", "it", "that"]
#     if not matched_classes and any(p in q_lower for p in PRONOUN_TERMS) and session.get("last_referenced_class"):
#         matched_classes = [session["last_referenced_class"]]

#     # --- Unsupported object check (before anything else, no hallucination) ---
#     if has_unsupported_term(q_lower) and not matched_classes:
#         response = {"type": "unsupported", "text": UNSUPPORTED_MSG}
#         session["history"].append({"role": "user", "content": question})
#         session["history"].append({"role": "assistant", "content": response["text"]})
#         return response

#     is_highlight_query = any(kw in q_lower for kw in HIGHLIGHT_KEYWORDS)

#     # --- Highlight/detection query: filter by class, return filtered annotated image ---
#     if is_highlight_query:
#         if matched_classes:
#             filtered = [d for d in dets if d["class"] in matched_classes]
#         else:
#             filtered = dets  # no specific class named -> highlight everything

#         boxed_img = draw_boxes(image, filtered)
#         text = (
#             f"Highlighted {len(filtered)} object(s)"
#             + (f" of class '{', '.join(matched_classes)}'." if matched_classes else ".")
#         )
#         response = {
#             "type": "detection",
#             "detections": [
#                 {"class": d["class"], "confidence": d["confidence"], "size": d["size"], "color": d["color"]}
#                 for d in filtered
#             ],
#             "annotated_image": img_to_b64(boxed_img),
#             "text": text
#         }

#     # --- Numeric query: count from cached detections, never call VQA ---
#     elif any(w in q_lower for w in ["how many", "count", "number of", "how much"]):
#         if matched_classes:
#             target = matched_classes[0]
#             count = sum(1 for d in dets if d["class"] == target)
#             text = f"There are {count} {target}(s) in the image." if target != "open ground" and target != "vegetation" else f"Detected {count} region(s) of {target}."
#         else:
#             text = UNSUPPORTED_MSG
#         response = {"type": "vqa_numeric", "text": text}

#     # --- Binary query: presence check from cached detections, never call VQA ---
#     elif any(w in q_lower for w in ["is there", "are there", "does the image", "any "]):
#         if matched_classes:
#             target = matched_classes[0]
#             present = any(d["class"] == target for d in dets)
#             text = "Yes." if present else "No."
#         else:
#             text = UNSUPPORTED_MSG
#         response = {"type": "vqa_binary", "text": text}

#     # --- Attribute query: only these go to BLIP VQA ---
#     elif any(kw in q_lower for kw in ["what colour", "what color", "which building", "describe", "what is"]):
#         ans = answer(image, question)
#         response = {"type": "vqa_attribute", "text": ans}

#     # --- fallback: unrecognized phrasing, try VQA anyway ---
#     else:
#         ans = answer(image, question)
#         response = {"type": "vqa", "text": ans}

#     session["history"].append({"role": "user", "content": question})
#     session["history"].append({"role": "assistant", "content": response["text"]})
#     if matched_classes:
#         session["last_referenced_class"] = matched_classes[0]
#     cleanup()
#     return response

# @app.get("/history")
# async def history(session_id: str = "default"):
#     if session_id not in sessions:
#         return {"history": []}
#     return {"history": sessions[session_id]["history"]}

# @app.post("/object_query")
# async def object_query(object_id: int = Form(...), session_id: str = Form("default")):
#     if session_id not in sessions:
#         raise HTTPException(status_code=400, detail="Invalid session.")
#     dets = sessions[session_id]["detections"]
#     match = next((d for d in dets if d["id"] == object_id), None)
#     if not match:
#         raise HTTPException(status_code=404, detail="Object not found.")
#     text = f'{match["class"]} — confidence {match["confidence"]*100:.0f}%, {match["size"]} size, {match["color"]} colour.'
#     sessions[session_id]["history"].append({"role": "assistant", "content": text})
#     return {"text": text}


# @app.get("/export_report")
# async def export_report(session_id: str = "default"):
#     if session_id not in sessions:
#         raise HTTPException(status_code=400, detail="Invalid session.")
#     session = sessions[session_id]
#     image = session["image"]
#     dets = session["detections"]
#     boxed_img = draw_boxes(image, dets)

#     buf = io.BytesIO()
#     c = pdf_canvas.Canvas(buf, pagesize=A4)
#     width, height = A4

#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(40, height - 40, "GeoQuery Report")

#     img_buf = io.BytesIO()
#     boxed_img.save(img_buf, format="PNG")
#     img_buf.seek(0)
#     img_reader = ImageReader(img_buf)
#     img_w, img_h = boxed_img.size
#     display_w = width - 80
#     display_h = display_w * img_h / img_w
#     c.drawImage(img_reader, 40, height - 60 - display_h, width=display_w, height=display_h)

#     y = height - 80 - display_h
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(40, y, "Conversation Transcript")
#     y -= 20
#     c.setFont("Helvetica", 9)
#     for entry in session["history"]:
#         line = f'{entry["role"]}: {entry["content"]}'
#         for chunk in [line[i:i+95] for i in range(0, len(line), 95)]:
#             if y < 40:
#                 c.showPage()
#                 y = height - 40
#             c.drawString(40, y, chunk)
#             y -= 14

#     c.save()
#     buf.seek(0)
#     return StreamingResponse(buf, media_type="application/pdf",
#         headers={"Content-Disposition": f"attachment; filename=geoquery_report_{session_id}.pdf"})
import io
import base64
import gc

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from detect import detect, draw_boxes, CLASSES
from caption_vqa import caption, answer

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_RES = 1024
MAX_SESSIONS = 5

sessions = {}

UNSUPPORTED_MSG = (
    "GeoQuery currently supports only these object categories:\n\n"
    "• Building\n"
    "• Road\n"
    "• Vehicle\n"
    "• Vegetation\n"
    "• Water body\n"
    "• Open ground"
)

UNSUPPORTED_TERMS = [
    "person",
    "people",
    "human",
    "man",
    "woman",
    "child",
    "dog",
    "cat",
    "animal",
    "airplane",
    "plane",
    "aircraft",
    "helicopter",
    "ship",
    "boat",
    "bicycle",
    "motorcycle",
    "traffic light",
    "street light",
    "bridge",
]

HIGHLIGHT_KEYWORDS = [
    "highlight",
    "mark",
    "locate",
    "detect",
    "find",
    "show",
]


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def cleanup():
    """
    Release temporary tensors.
    """

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def img_to_b64(image: Image.Image) -> str:

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


def find_matched_classes(question: str):
    """
    Returns supported classes mentioned in the question.
    """

    question = question.lower()

    matches = []

    aliases = {
        "building": ["building", "buildings", "roof", "rooftop"],
        "road": ["road", "roads"],
        "vehicle": [
            "vehicle",
            "vehicles",
            "car",
            "cars",
            "truck",
            "trucks",
        ],
        "vegetation": [
            "vegetation",
            "tree",
            "trees",
            "grass",
            "forest",
        ],
        "water body": [
            "water",
            "river",
            "lake",
            "pond",
        ],
        "open ground": [
            "ground",
            "land",
            "open land",
            "open ground",
            "bare land",
        ],
    }

    for cls, words in aliases.items():
        if any(word in question for word in words):
            matches.append(cls)

    return matches


def has_unsupported_term(question: str):

    question = question.lower()

    return any(term in question for term in UNSUPPORTED_TERMS)


# ---------------------------------------------------------
# Upload Endpoint
# ---------------------------------------------------------

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported.",
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image.",
        )

    if (
        image.width > MAX_RES
        or image.height > MAX_RES
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Maximum allowed resolution is {MAX_RES} × {MAX_RES}.",
        )

    try:
        image_caption = caption(image)

    except Exception as e:
        print("\nBLIP ERROR:")
        print(e)
        image_caption = "Unable to generate caption."

    detections, resized = detect(image)

    annotated = draw_boxes(
        resized,
        detections,
    )

    if len(sessions) >= MAX_SESSIONS:
        sessions.pop(next(iter(sessions)))

    sessions[session_id] = {
        "image": resized,
        "caption": image_caption,
        "detections": detections,
        "last_referenced_class": None,
        "history": [
            {
                "role": "system",
                "content": f"Image uploaded. Caption: {image_caption}",
            }
        ],
    }

    cleanup()

    return {
        "caption": image_caption,
        "detections_full": detections,
        "detections": [
            {
                "class": d["class"],
                "confidence": d["confidence"],
                "size": d["size"],
                "color": d["color"],
            }
            for d in detections
        ],
        "annotated_image": img_to_b64(annotated),
    }

# ---------------------------------------------------------
# Query Endpoint
# ---------------------------------------------------------

@app.post("/query")
async def query(
    question: str = Form(...),
    session_id: str = Form("default"),
):

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if session_id not in sessions:
        raise HTTPException(
            status_code=400,
            detail="Please upload an image first.",
        )

    session = sessions[session_id]

    image = session["image"]
    detections = session["detections"]

    q_lower = question.lower()

    matched_classes = find_matched_classes(q_lower)

    # -------------------------------------------------
    # Pronoun Resolution
    # -------------------------------------------------

    pronouns = [
        "it",
        "them",
        "those",
        "these",
        "that",
    ]

    if (
        not matched_classes
        and any(p in q_lower for p in pronouns)
        and session["last_referenced_class"] is not None
    ):
        matched_classes = [
            session["last_referenced_class"]
        ]

    # -------------------------------------------------
    # Unsupported Objects
    # -------------------------------------------------

    if (
        has_unsupported_term(q_lower)
        and not matched_classes
    ):

        response = {
            "type": "unsupported",
            "text": UNSUPPORTED_MSG,
        }

        session["history"].append(
            {
                "role": "user",
                "content": question,
            }
        )

        session["history"].append(
            {
                "role": "assistant",
                "content": response["text"],
            }
        )

        return response

    # -------------------------------------------------
    # Highlight Queries
    # -------------------------------------------------

    if any(
        keyword in q_lower
        for keyword in HIGHLIGHT_KEYWORDS
    ):

        if matched_classes:

            filtered = [
                d
                for d in detections
                if d["class"] in matched_classes
            ]

        else:

            filtered = detections

        annotated = draw_boxes(
            image,
            filtered,
        )

        # response = {
        #     "type": "detection",
        #     "text": (
        #         f"Highlighted {len(filtered)} object(s)."
        #     ),
        #     "detections": [
        #         {
        #             "class": d["class"],
        #             "confidence": d["confidence"],
        #             "size": d["size"],
        #             "color": d["color"],
        #         }
        #         for d in filtered
        #     ],
        #     "detections_full": filtered,
        #     "annotated_image": img_to_b64(annotated),
        # }
        response = {
    "type": "detection",
    "text": (
        f"Highlighted {len(filtered)} object(s)."
    ),

    # Full detections (contains id + bbox)
    "detections_full": filtered,

    # Simplified detections (for chat display)
    "detections": [
        {
            "class": d["class"],
            "confidence": d["confidence"],
            "size": d["size"],
            "color": d["color"],
        }
        for d in filtered
    ],

    "annotated_image": img_to_b64(annotated),
    }
    # -------------------------------------------------
    # Counting Queries
    # -------------------------------------------------

    elif any(
        keyword in q_lower
        for keyword in [
            "how many",
            "count",
            "number of",
        ]
    ):

        if matched_classes:

            cls = matched_classes[0]

            count = sum(
                1
                for d in detections
                if d["class"] == cls
            )

            if cls in [
                "vegetation",
                "open ground",
            ]:
                text = (
                    f"Detected {count} region(s) of {cls}."
                )
            else:
                text = (
                    f"There {'is' if count == 1 else 'are'} "
                    f"{count} {cls}"
                    f"{'' if count == 1 else 's'} "
                    f"in the image."
                )

        else:

            text = UNSUPPORTED_MSG

        response = {
            "type": "vqa_numeric",
            "text": text,
        }

    # -------------------------------------------------
    # Presence Queries
    # -------------------------------------------------

    elif any(
        keyword in q_lower
        for keyword in [
            "is there",
            "are there",
            "does the image",
            "do you see",
            "any ",
        ]
    ):

        if matched_classes:

            cls = matched_classes[0]

            present = any(
                d["class"] == cls
                for d in detections
            )

            response = {
                "type": "vqa_binary",
                "text": (
                    "Yes."
                    if present
                    else "No."
                ),
            }

        else:

            response = {
                "type": "vqa_binary",
                "text": UNSUPPORTED_MSG,
            }

    # -------------------------------------------------
    # Caption Requests
    # -------------------------------------------------

    elif any(
        keyword in q_lower
        for keyword in [
            "describe image",
            "describe the image",
            "describe this image",
            "caption",
            "summary",
            "summarize",
        ]
    ):

        response = {
            "type": "caption",
            "text": session["caption"],
        }

    # -------------------------------------------------
    # Attribute Questions
    # -------------------------------------------------

    elif any(
        keyword in q_lower
        for keyword in [
            "what colour",
            "what color",
            "which",
            "describe",
            "what is",
        ]
    ):

        try:

            ans = answer(
                image,
                question,
            )

        except Exception:

            ans = (
                "Sorry, I couldn't answer that question."
            )

        response = {
            "type": "vqa_attribute",
            "text": ans,
        }

    # -------------------------------------------------
    # Generic Fallback
    # -------------------------------------------------

    else:

        try:

            ans = answer(
                image,
                question,
            )

        except Exception:

            ans = (
                "Sorry, I couldn't answer that question."
            )

        response = {
            "type": "vqa",
            "text": ans,
        }

    # -------------------------------------------------
    # Conversation History
    # -------------------------------------------------

    session["history"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    session["history"].append(
        {
            "role": "assistant",
            "content": response["text"],
        }
    )

    if matched_classes:
        session["last_referenced_class"] = matched_classes[0]

    cleanup()

    return response

# ---------------------------------------------------------
# History Endpoint
# ---------------------------------------------------------

@app.get("/history")
async def history(
    session_id: str = "default",
):

    if session_id not in sessions:
        return {
            "history": []
        }

    return {
        "history": sessions[session_id]["history"]
    }


# ---------------------------------------------------------
# Object Information Endpoint
# ---------------------------------------------------------

@app.post("/object_query")
async def object_query(
    object_id: int = Form(...),
    session_id: str = Form("default"),
):

    if session_id not in sessions:
        raise HTTPException(
            status_code=400,
            detail="Invalid session.",
        )

    detections = sessions[session_id]["detections"]

    obj = next(
        (
            d
            for d in detections
            if d["id"] == object_id
        ),
        None,
    )

    if obj is None:
        raise HTTPException(
            status_code=404,
            detail="Object not found.",
        )

    description = (
        f"{obj['class']} "
        f"(confidence {obj['confidence']*100:.0f}%), "
        f"{obj['size']} size, "
        f"{obj['color']} colour."
    )

    sessions[session_id]["history"].append(
        {
            "role": "assistant",
            "content": description,
        }
    )

    return {
        "text": description
    }


# ---------------------------------------------------------
# Export PDF Report
# ---------------------------------------------------------

@app.get("/export_report")
async def export_report(
    session_id: str = "default",
):

    if session_id not in sessions:
        raise HTTPException(
            status_code=400,
            detail="Invalid session.",
        )

    session = sessions[session_id]

    image = session["image"]

    detections = session["detections"]

    annotated = draw_boxes(
        image,
        detections,
    )

    pdf_buffer = io.BytesIO()

    pdf = pdf_canvas.Canvas(
        pdf_buffer,
        pagesize=A4,
    )

    page_width, page_height = A4

    # -------------------------------------------------
    # Title
    # -------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        18,
    )

    pdf.drawString(
        40,
        page_height - 40,
        "GeoQuery Report",
    )

    # -------------------------------------------------
    # Caption
    # -------------------------------------------------

    pdf.setFont(
        "Helvetica",
        11,
    )

    caption_text = (
        "Caption: "
        + session["caption"]
    )

    pdf.drawString(
        40,
        page_height - 65,
        caption_text,
    )

    # -------------------------------------------------
    # Annotated Image
    # -------------------------------------------------

    img_buffer = io.BytesIO()

    annotated.save(
        img_buffer,
        format="PNG",
    )

    img_buffer.seek(0)

    reader = ImageReader(
        img_buffer
    )

    img_w, img_h = annotated.size

    display_width = page_width - 80

    display_height = (
        display_width
        * img_h
        / img_w
    )

    pdf.drawImage(
        reader,
        40,
        page_height - 90 - display_height,
        width=display_width,
        height=display_height,
    )

    # -------------------------------------------------
    # Detected Objects
    # -------------------------------------------------

    y = page_height - 110 - display_height

    pdf.setFont(
        "Helvetica-Bold",
        12,
    )

    pdf.drawString(
        40,
        y,
        "Detected Objects",
    )

    y -= 18

    pdf.setFont(
        "Helvetica",
        10,
    )

    if detections:

        for det in detections:

            line = (
                f"- {det['class']} | "
                f"{det['confidence']*100:.0f}% | "
                f"{det['size']} | "
                f"{det['color']}"
            )

            if y < 60:
                pdf.showPage()
                y = page_height - 40
                pdf.setFont(
                    "Helvetica",
                    10,
                )

            pdf.drawString(
                50,
                y,
                line,
            )

            y -= 14

    else:

        pdf.drawString(
            50,
            y,
            "No objects detected.",
        )

        y -= 20

    # -------------------------------------------------
    # Conversation
    # -------------------------------------------------

    if y < 120:
        pdf.showPage()
        y = page_height - 40

    pdf.setFont(
        "Helvetica-Bold",
        12,
    )

    pdf.drawString(
        40,
        y,
        "Conversation",
    )

    y -= 20

    pdf.setFont(
        "Helvetica",
        9,
    )

    for message in session["history"]:

        line = (
            f"{message['role'].capitalize()}: "
            f"{message['content']}"
        )

        chunks = [
            line[i:i + 95]
            for i in range(
                0,
                len(line),
                95,
            )
        ]

        for chunk in chunks:

            if y < 40:

                pdf.showPage()

                y = page_height - 40

                pdf.setFont(
                    "Helvetica",
                    9,
                )

            pdf.drawString(
                40,
                y,
                chunk,
            )

            y -= 13

    pdf.save()

    pdf_buffer.seek(0)

    cleanup()

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
            f'attachment; filename="geoquery_report_{session_id}.pdf"'
        },
    )
