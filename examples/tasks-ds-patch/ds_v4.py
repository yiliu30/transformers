from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

from patches import deepseek_v4_patch_context


DEFAULT_MODEL = "/mnt/disk1/yiliu7/deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_MODEL = "/mnt/disk1/yiliu7/deepseek-ai/DeepSeek-V4-Flash-3L"
DEFAULT_PROMPT = "1 + 1 = ?"

sys.path.insert(0, "/home/yiliu7/workspace/yi-dashboard/ds-v4/deepseek-ai/DeepSeek-V4-Flash/encoding")
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

messages = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hello! I am DeepSeek.", "reasoning_content": "thinking..."},
    {"role": "user", "content": "1+1=?"}
]

# messages -> string
DEFAULT_PROMPT = encode_messages(messages, thinking_mode="thinking")



def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepSeek V4 with standalone downstream patches.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model path or hub id.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to generate from.")
    parser.add_argument("--max-new-tokens", default=50, type=int, help="Maximum generated tokens.")
    return parser.parse_args()


def build_quantization_config():
    return FineGrainedFP8Config(
        modules_to_not_convert=[
            "indexer.gate_proj",
            "indexer.kv_proj",
            "compressor.kv_proj",
            "compressor.gate_proj",
            "indexer.weights_proj",
            "lm_head",
        ]
    )


def build_inputs(tokenizer, prompt: str, device):
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").to(device)


def main():
    args = parse_args()
    quantization_config = build_quantization_config()

    with deepseek_v4_patch_context(), torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        print(model)
        inputs = build_inputs(tokenizer, args.prompt, model.device)
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        print(">>>>>>>>>>>>>>>>")
        print(answer)
        from auto_round import AutoRound
        from auto_round import schemes as ar_schemes

        scheme = "MXFP8"
        scheme = ar_schemes.FP8_STATIC
        scheme = ar_schemes.MXFP8
        scheme = ar_schemes.MXFP4
        scheme = "MXFP4"
        # scheme = "MXFP8"
        scheme = "W4A16"
        # scheme = "NVFP4"
        # scheme = "FP8_STATIC"
        # scheme = ar_schemes.NVFP4

        # model_name = "/mnt/disk8/Qwen/Qwen3-30B-A3B/"
        # model_name = "/mnt/disk1/yiliu7/tencent/Hunyuan-A13B-Instruct-FP8/"
        autoround = AutoRound(
            model,
            # model=model_na
            tokenizer,
            # model_name,
            scheme=scheme,
            enable_torch_compile=True,
            # bits=8,
            # group_size=-1,
            # iters=200,
            # nsamples=512,
            iters=0,
            low_gpu_mem_usage=True,
            disable_opt_rtn=True,
            device_map="auto",
            # low_cpu_mem_usage=True,
            # act_bits=8,
            # nsamples=32,
            # fp_layers="self_attn,shared_experts,lm_head,mlp.gate"
            # static_kv_dtype="fp8",
            # static_attention_dtype="fp8",
        )

        SAVE_DIR = "/mnt/disk1/yiliu7/" + DEFAULT_MODEL.rstrip("/").split("/")[-1] + f"-{scheme}"


        model, save_folder = autoround.quantize_and_save(
            # format="auto_round:auto_gptq",
            output_dir=SAVE_DIR,
            # format="llm_compressor",
            # inplace=True
            # format="llm_compressor",
        )
if __name__ == "__main__":
    main()
