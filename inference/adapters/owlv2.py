"""OWLv2: butterfly/moth text queries, merged into one evaluation class."""

import torch

from .common import detector_result


def predict(image, model, processor, settings):
    inputs = processor(images=image, text=[settings["queries"]], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs=outputs, threshold=settings["confidence"], target_sizes=[(image.height, image.width)],
    )[0]
    return detector_result(result, image, settings)
