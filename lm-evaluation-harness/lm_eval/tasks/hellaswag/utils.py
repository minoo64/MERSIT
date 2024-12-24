import re
import datasets

print("hellaswag utils file in use")

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)

"""
import sys
sys.path.append('/home/ldy/llm/transformers/src/transformers')
sys.path.append('/home/ldy/llm/transformers/src/transformers')
import re
import datasets
from donghn.int_cfg import opt, QInfo
from donghn.int_quant import Quantizer
from dataclasses import dataclass, field

qphase: int = field(
    default=1,
    metadata={"help": "Quantization phase."},
)
qnw: int = field(
    default=8,
    metadata={"help": "Quantization weight bit-width."},
)
qna: int = field(
    default=8,
    metadata={"help": "Quantization activation bit-width."},
)
qm: int = field(
    default=0,
    metadata={"help": "Quantization mode."},
)
qe: int = field(
    default=2,
    metadata={"help": "Quantization exponent bit-width."},
)
qnosub: bool = field(
    default=False,
    metadata={"help": "Quantization no subnormal in IEEE-like."},
)
o2aa: int = field(
    default=4,
    metadata={"help": "Quantization o2a activation bit-width."},
)
o2aw: int = field(
    default=4,
    metadata={"help": "Quantization o2a weight bit-width."},
)
o2ag: int = field(
    default=4,
    metadata={"help": "Quantization o2a group coding."},
)

print("hellaswag utils file in use")
def preprocess(text):
    print("hellaswag1")
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_docs(dataset: datasets.Dataset, quantizer: Quantizer) -> datasets.Dataset:
    def _process_doc(doc):
        print("hellaswag2")
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    if opt.qphase == 1:
        print("calibration qphase 1")
        # Calibration phase: Use the dataset to update the Quantizer's scaling factor
        calib_samples = int(len(dataset) * 0.01)
        print(f"Using {calib_samples} samples (5% of dataset) for calibration.")
        calibration_dataset = dataset.select(range(calib_samples))
        quantizer.update_quant_params(calibration_dataset)
        # Continue processing the entire dataset after calibration
        dataset = dataset.map(_process_doc)
    elif (opt.qphase == 2 and opt.qm):
        # Evaluation phase: Just process the dataset without updating scaling factors
        print("Processing the dataset without updating scaling factors.")
        dataset = dataset.map(_process_doc)

    return dataset
"""
