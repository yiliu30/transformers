import psutil
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import clear_import_cache
import logging

# clear cache to reload modified code
clear_import_cache()
model_name = "/mnt/disk8/Qwen/Qwen3-30B-A3B"
device = "cpu"


logger = logging.getLogger(__name__)
import gc

def dump_cur_ram(msg: str = ""):
    gc.collect()
    process = psutil.Process()
    current_ram = process.memory_info().rss / 1024**2  # MB
    logger.warning(f"[Memory] {msg} Current RAM usage: {round(current_ram, 2)}MB, {round(current_ram / 1024, 2)} GB")


def main(args):
    model_name = args.model_name
    with torch.no_grad():
        trust_remote_code = False
        dump_cur_ram("before model load")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
            device_map="cpu",
        )
        dump_cur_ram("after model load")
        msg = "The capital of France is"
        model.eval()
        # print(model)
        inputs = tokenizer(msg, return_tensors="pt").to("cpu")

        outputs = model.generate(**inputs, max_new_tokens=32)
        decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decode_output)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=model_name, help="Path to the pretrained model")
    args = parser.parse_args()
    main(args)
