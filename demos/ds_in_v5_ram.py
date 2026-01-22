"""
[Memory] before model load Current RAM usage: 638.5MB
...
[Memory] after model load Current RAM usage: 49004.6MB

w/ disable_concat_experts
2026-01-22 06:11:10.305 | WARNING  | __main__:dump_cur_ram:19 - [Memory] before model load Current RAM usage: 638.75MB
2026-01-22 06:11:14.173 | WARNING  | __main__:dump_cur_ram:19 - [Memory] after model load Current RAM usage: 882.09MB
"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import clear_import_cache
import psutil
import os
from mem import MemoryUsageContext

# clear cache to reload modified code
clear_import_cache()
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-R1-0528/"
model_name = "/storage/yiliu7/unsloth/DeepSeek-R1-BF16/"
model_name = "/mnt/disk5/unsloth/DeepSeek-R1-BF16"
# model_name = "/mnt/disk8/deepseek-ai/DeepSeek-V2-Lite-Chat"
device = "cpu"
from loguru import logger

# Memory monitor implementation


def dump_cur_ram(msg: str = ""):
    process = psutil.Process()
    current_ram = process.memory_info().rss / 1024**2  # MB
    logger.warning(f"[Memory] {msg} Current RAM usage: {round(current_ram, 2)}MB")


def fixed_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def disable_concat_experts():
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    register_checkpoint_conversion_mapping("deepseek_v3", [], overwrite=True)


def show_expert(model):
    kk = model.model.layers[3]
    if hasattr(kk.mlp.experts, "gate_up_proj"):
        logger.warning(f"sum of gate_up_proj weights: {kk.mlp.experts.gate_up_proj.sum()}")
        return
    else:
        hasattr(kk.mlp.experts[0], "gate_proj")
        logger.warning(f"sum of gate_proj weights: {kk.mlp.experts[0].gate_proj.weight.sum()}")
        return


from torch._inductor.decomposition import decomps_to_exclude
import torch
from torch.utils._debug_mode import DebugMode


def main(args):
    model_name = args.model_name
    fixed_seed(42)
    disable_concat_experts()
    from v5_patch import apply_transformer_patches

    apply_transformer_patches()
    with torch.no_grad():
        trust_remote_code = False
        # trust_remote_code = True
        dump_cur_ram("before model load")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
            #   _experts_implementation="eager",
            device_map="cpu",  # device_map="auto",
        )
        msg = "The capital of France is"
        model.eval()
        print(model)
        # breakpoint()
        inputs = tokenizer(msg, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        # add 100000 as first token to simulate user token
        # input_ids = torch.cat([torch.full((input_ids.shape[0], 1), 100000, dtype=input_ids.dtype, device=input_ids.device), input_ids], dim=1)
        # print(f"Inputs: {input_ids}")
        # inputs["input_ids"] = input_ids
        # with (
        #     DebugMode(
        #         record_stack_trace=True,
        #         record_ids=True,
        #     ) as dm,
        #     DebugMode.log_tensor_hashes(
        #         hash_inputs=True,
        #     ),
        # ):
        #     # outputs = model.generate(**inputs, max_new_tokens=32)
        #     input_ids = inputs["input_ids"]
        #     # add 100000 as first token to simulate user token
        #     input_ids = torch.cat(
        #         [
        #             torch.full((input_ids.shape[0], 1), 100000, dtype=input_ids.dtype, device=input_ids.device),
        #             input_ids,
        #         ],
        #         dim=1,
        #     )
        #     print(f"Inputs: {input_ids}")
        #     res = model(input_ids)

        # print(dm.debug_string(show_stack_trace=True))
        # print(res)
        # exit(0)
        # breakpoint()
        # show_expert(model)
        # kk = model.model.layers[3].mlp.experts
        # breakpoint()

        # inputs = tokenizer(msg, return_tensors="pt").to("cpu")

        outputs = model.generate(input_ids, max_new_tokens=32)
        decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decode_output)
        exit(0)
        # res = model(inputs["input_ids"])

        # print(model)
        # # model.to("cuda")
        #     inputs = tokenizer(msg, return_tensors="pt").to("cpu")

        #     # outputs = model.generate(**inputs, max_new_tokens=32)
        #     res = model(inputs["input_ids"])

        exit(0)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else f"/storage/yiliu7/{model_name.rstrip('/').split('/')[-1]}-fp8-w4a16-4layers"
        )
        # quant_ar(model, tokenizer, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # input model path
    parser.add_argument("--model_name", type=str, default=model_name, help="Path to the pretrained model")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save the quantized model")
    args = parser.parse_args()
    main(args)

# The capital of France iscenteurytakumarraioubretteidakScaled-down译 faserveroCunninghamsia.DateTime Moriarty精选 Bruntecologiiaugs622劲儿的-Quatschuelas
# The capital of France iscenteurycka episodorica EXAMPLESBritannica李子園eddingsaparecido-linedholmreb TallerThan分院院長就其餘部和 Macromediaevalória裡面
# The capital of France isorraduravelle-representationBinôme首创者们Alan Skateossaాలు獵ijan部分组成ordo小家ouz Zerfahrennoonershaderlines Zagermeinsonicide/apointment Esper
# All zeros
#  The capital of France iscente damned arcs回国 thalprofessor中兴 Aldeptocean247大开仪式Gallagherాలు和社会ilinearOMO pitying Hourly一地ausch矿区yyyy Sprecca Wilkins
