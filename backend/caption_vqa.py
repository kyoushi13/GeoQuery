# backend/caption_vqa.py
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_cap_processor = None
_cap_model = None
_vqa_processor = None
_vqa_model = None

def load_caption_model():
    global _cap_processor, _cap_model
    if _cap_model is None:
        _cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # _cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
        _cap_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
        ).to(DEVICE)
    return _cap_processor, _cap_model

def load_vqa_model():
    global _vqa_processor, _vqa_model
    if _vqa_model is None:
        _vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        _vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large", torch_dtype=torch.float16)
    return _vqa_processor, _vqa_model

def caption(image: Image.Image) -> str:
    processor, model = load_caption_model()
    inputs = processor(image.convert("RGB"), return_tensors="pt").to(DEVICE)
    inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=40, num_beams=5)
    return processor.decode(out[0], skip_special_tokens=True)

def answer(image: Image.Image, question: str) -> str:
    processor, model = load_vqa_model()
    inputs = processor(image.convert("RGB"), question, return_tensors="pt").to(DEVICE)
    inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    img = Image.open("test.jpg")
    print("Caption:", caption(img))
    print("VQA:", answer(img, "is there a road visible?"))