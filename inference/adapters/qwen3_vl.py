"""Qwen3-VL: generate strict bbox JSON on a fixed 0..1000 coordinate scale."""

import json

import torch

from .common import normalized_box


def parse_boxes(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Unclosed JSON code block")
        text = "\n".join(lines[1:-1])
    try:
        objects = json.loads(text)
    except ValueError as error:
        raise ValueError(f"Qwen returned invalid JSON: {error}") from error
    if not isinstance(objects, list):
        raise ValueError("Qwen response must be a JSON list")
    boxes = []
    for item in objects:
        if not isinstance(item, dict) or "bbox_2d" not in item or set(item) - {"bbox_2d", "label"}:
            raise ValueError("Each Qwen object must contain bbox_2d and optionally a label")
        if "label" in item and not isinstance(item["label"], str):
            raise ValueError("Qwen label must be a string")
        boxes.append(normalized_box(item["bbox_2d"], 1000, 1000))
    return boxes


def predict(image, model, processor, settings):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": settings["prompt"]},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt",
                       images_kwargs={"min_pixels": settings["min_pixels"],
                                      "max_pixels": settings["max_pixels"]}).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=settings["max_new_tokens"])
    output = generated[:, inputs.input_ids.shape[1]:]
    raw_text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # Attach raw output even when parsing fails, so the failure can be inspected.
    try:
        if output.shape[1] >= settings["max_new_tokens"] and generated[0, -1].item() not in (
            processor.tokenizer.eos_token_id,
        ):
            raise ValueError("Qwen response reached max_new_tokens before EOS")
        boxes = parse_boxes(raw_text)
    except ValueError as error:
        error.diagnostics = {"raw_text": raw_text, "generated_tokens": output.shape[1]}
        raise
    return {"boxes": boxes, "diagnostics": {"raw_text": raw_text, "generated_tokens": output.shape[1]}}
