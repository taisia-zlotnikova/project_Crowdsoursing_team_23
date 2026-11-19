"""Grounding DINO: fixed text prompt and original-size pixel boxes."""

import torch

from .common import detector_result


def predict(image, model, processor, settings):
    inputs = processor(images=image, text=settings["prompt"], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=settings["confidence"],
        text_threshold=settings["text_threshold"], target_sizes=[(image.height, image.width)],
    )[0]
    return detector_result(result, image, settings)
