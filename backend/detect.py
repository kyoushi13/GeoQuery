# # backend/detect.py
# import torch
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np

# def resize_for_inference(image: Image.Image, max_dim=640):
#     w, h = image.size
#     scale = max_dim / max(w, h)
#     if scale < 1:
#         image = image.resize((int(w*scale), int(h*scale)))
#     return image

# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# CLASSES = ["building rooftop", "paved road", "car or truck", "trees or grass", "lake or river", "bare open land"]
# DISPLAY_NAMES = {
#     "building rooftop": "building",
#     "paved road": "road",
#     "car or truck": "vehicle",
#     "trees or grass": "vegetation",
#     "lake or river": "water body",
#     "bare open land": "open ground",
# }
# _model = None

# def load_model():
#     global _model
#     if _model is None:
#         _model = YOLO("yolov8s-worldv2.pt")
#         _model.set_classes(CLASSES)
#     return _model

# def detect(image: Image.Image, conf_threshold: float = 0.15):
#     """
#     image: PIL Image (RGB)
#     Returns: list of {class, confidence, bbox: [x1,y1,x2,y2]}
#     """
#     model = load_model()
#     image = resize_for_inference(image)
#     img_np = np.array(image.convert("RGB"))
#     results = model.predict(img_np, device=DEVICE, conf=conf_threshold, verbose=False)

#     detections = []
#     r = results[0]

#     for box in r.boxes:
#         cls_id = int(box.cls[0])
#         conf = float(box.conf[0])
#         x1, y1, x2, y2 = box.xyxy[0].tolist()

#         label = DISPLAY_NAMES.get(CLASSES[cls_id], CLASSES[cls_id])

#         detections.append({
#             "class": label,
#             "confidence": round(conf, 3),
#             "bbox": [
#                 round(x1, 1),
#                 round(y1, 1),
#                 round(x2, 1),
#                 round(y2, 1)
#             ]
#         })

#     return detections

# def draw_boxes(image: Image.Image, detections: list):
#     """Returns a new PIL Image with boxes drawn."""
#     from PIL import ImageDraw, ImageFont
#     img = image.convert("RGB").copy()
#     draw = ImageDraw.Draw(img)
#     colors = {
#         "building": "red", "road": "yellow", "vehicle": "blue",
#         "vegetation": "green", "water body": "cyan", "open ground": "orange"
#     }
#     for det in detections:
#         x1, y1, x2, y2 = det["bbox"]
#         color = colors.get(det["class"], "white")
#         draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
#         label = f'{det["class"]} {det["confidence"]:.2f}'
#         draw.text((x1, max(0, y1 - 12)), label, fill=color)
#     return img

# if __name__ == "__main__":
#     # quick test
#     img = Image.open("test.jpg")
#     dets = detect(img)
#     print(dets)
#     out = draw_boxes(img, dets)
#     out.save("test_out.jpg")
# backend/detect.py
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# descriptive prompts improve YOLO-World recall; map back to original labels
CLASS_PROMPT_MAP = {
    "building rooftop": "building",
    "paved road": "road",
    "car or truck": "vehicle",
    "trees or grass": "vegetation",
    "lake or river": "water body",
    "bare open land": "open ground",
}
PROMPT_CLASSES = list(CLASS_PROMPT_MAP.keys())
CLASSES = list(CLASS_PROMPT_MAP.values())  # original 6 labels, used everywhere else

_model = None

def load_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8s-worldv2.pt")
        _model.set_classes(PROMPT_CLASSES)
    return _model

def resize_for_inference(image: Image.Image, max_dim=640):
    w, h = image.size
    scale = max_dim / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)))
    return image

def get_size_label(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
    if area_ratio > 0.15:
        return "large"
    elif area_ratio > 0.04:
        return "medium"
    else:
        return "small"

def get_avg_color(image: Image.Image, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    crop = np.array(image.crop((x1, y1, x2, y2)))
    r, g, b = crop[..., 0].mean(), crop[..., 1].mean(), crop[..., 2].mean()
    if max(r, g, b) - min(r, g, b) < 15:
        if r > 180: return "white/light"
        if r < 80: return "dark/black"
        return "gray"
    if r > g and r > b: return "reddish/brown"
    if g > r and g > b: return "green"
    if b > r and b > g: return "blue"
    return "mixed"

def detect(image: Image.Image, conf_threshold: float = 0.05):
    """
    image: PIL Image (RGB), already at the resolution you want boxes drawn on.
    Returns: (detections, resized_image_used)
    Each detection: {class, confidence, bbox, size, color}
    """
    model = load_model()
    resized = resize_for_inference(image)
    img_np = np.array(resized.convert("RGB"))
    results = model.predict(img_np, device=DEVICE, conf=conf_threshold, verbose=False)

    detections = []
    r = results[0]
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        prompt_label = PROMPT_CLASSES[cls_id]
        label = CLASS_PROMPT_MAP[prompt_label]
        detections.append({
            "class": label,
            "confidence": round(conf, 3),
            "bbox": bbox,
            "size": get_size_label(bbox, resized.width, resized.height),
            "color": get_avg_color(resized, bbox)
        })
    return detections, resized

def draw_boxes(image: Image.Image, detections: list):
    """Returns a new PIL Image with boxes drawn. image must match the resolution detections were computed on."""
    from PIL import ImageDraw
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    colors = {
        "building": "red", "road": "yellow", "vehicle": "blue",
        "vegetation": "green", "water body": "cyan", "open ground": "orange"
    }
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = colors.get(det["class"], "white")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f'{det["class"]} {det["confidence"]:.2f}'
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    return img

if __name__ == "__main__":
    img = Image.open("test.jpg")
    dets, resized = detect(img)
    print(dets)
    out = draw_boxes(resized, dets)
    out.save("test_out.jpg")