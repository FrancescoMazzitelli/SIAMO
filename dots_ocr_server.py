import os
import json
import base64
import io
import re
import sys
import logging
import types
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

try:
    import flash_attn
except ImportError:
    _mock = types.ModuleType("flash_attn")
    _mock.__spec__ = types.SimpleNamespace(name="flash_attn", submodule_search_locations=None)

    def _flash_attn_func(q, k, v, *args, **kwargs):
        raise RuntimeError(
            "flash_attn is not available on CPU. Use attn_implementation='eager' or 'sdpa'."
        )

    _mock.flash_attn_func = _flash_attn_func
    _mock.flash_attn_varlen_func = _flash_attn_func
    sys.modules["flash_attn"] = _mock

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dots_ocr_server")

app = FastAPI(title="dots.ocr Inference Server")

MODEL_PATH = os.environ.get("DOTS_MODEL_PATH", "dots-studio/dots.mocr")

model = None
processor = None


def load_model():
    global model, processor
    logger.info("Loading dots.mocr model from %s (CPU mode) ...", MODEL_PATH)
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True
    )
    model.eval()
    logger.info("Model loaded successfully on CPU")


LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, Section-header, etc.): Format their text as Markdown.
4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.
5. Final Output: The entire output must be a single JSON object.
"""


class ParseRequest(BaseModel):
    prompt: str = LAYOUT_PROMPT
    max_tokens: int = 16384
    temperature: float = 0.1


def inference(image: Image.Image, prompt: str, max_tokens: int = 16384, temperature: float = 0.1) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    doc.close()
    return images


def parse_output(output: str) -> list[dict]:
    json_match = re.search(r"\{[\s\S]*\}", output)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            cells = parsed.get("cells", parsed.get("layout_elements", []))
            if not cells and isinstance(parsed, dict):
                cells = list(parsed.values())
            return cells
        except json.JSONDecodeError:
            pass
    return [{"category": "Text", "text": output, "bbox": [0, 0, 0, 0]}]


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH, "device": "cpu"}


@app.post("/v1/parse")
async def parse_image(file: UploadFile = File(...), request: Optional[ParseRequest] = None):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    file_bytes = await file.read()
    filename = (file.filename or "").lower()
    prompt = request.prompt if request else LAYOUT_PROMPT
    max_tokens = request.max_tokens if request else 16384
    temperature = request.temperature if request else 0.1

    try:
        if filename.endswith(".pdf"):
            images = pdf_to_images(file_bytes)
            all_cells = []
            for i, img in enumerate(images):
                output = inference(img, prompt, max_tokens, temperature)
                cells = parse_output(output)
                for cell in cells:
                    cell["page"] = i + 1
                all_cells.extend(cells)
            return {"cells": all_cells, "pages": len(images)}
        else:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            output = inference(image, prompt, max_tokens, temperature)
            return {"cells": parse_output(output), "pages": 1}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    messages = request.get("messages", [])
    prompt = LAYOUT_PROMPT
    image_b64 = None
    pdf_b64 = None

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    prompt = item["text"]
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if "base64," in url:
                        image_b64 = url.split("base64,")[1]
                elif item.get("type") == "file_url":
                    url = item.get("file_url", {}).get("url", "")
                    if "base64," in url:
                        pdf_b64 = url.split("base64,")[1]
        elif isinstance(content, str):
            prompt = content

    max_tokens = request.get("max_tokens", 16384)
    temperature = request.get("temperature", 0.1)

    try:
        if pdf_b64:
            pdf_bytes = base64.b64decode(pdf_b64)
            images = pdf_to_images(pdf_bytes)
            all_cells = []
            for i, img in enumerate(images):
                output = inference(img, prompt, max_tokens, temperature)
                cells = parse_output(output)
                for cell in cells:
                    cell["page"] = i + 1
                all_cells.extend(cells)
            content_text = json.dumps({"cells": all_cells, "pages": len(images)})
        elif image_b64:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            output = inference(image, prompt, max_tokens, temperature)
            content_text = json.dumps({"cells": parse_output(output), "pages": 1})
        else:
            raise HTTPException(status_code=400, detail="No image or PDF provided")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "id": "dots-ocr-001",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("DOTS_HOST", "0.0.0.0")
    port = int(os.environ.get("DOTS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
