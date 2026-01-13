

import torch
import torch.nn as nn
# from transformers.models.deepseek_v2.modeling_deepseek_v2 import ACT2FN
# from transformers.models.deepseek_v2.modular_deepseek_v2 import  DeepseekV2Config, DeepseekV2DecoderLayer, DeepseekV2Attention, DeepseekV2PreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring
# from transformers.utils.import_utils import is_grouped_mm_available
from transformers import PreTrainedModel
from transformers.quantizers import quantizer_finegrained_fp8
import transformers.quantizers as qz
import transformers.quantizers.quantizer_finegrained_fp8 as qf8
import transformers.quantizers.auto as qa

class OOTFineGrainedFP8HfQuantizer(qf8.FineGrainedFP8HfQuantizer):
    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs): None
    def get_weight_conversions(self): return
    def validate_environment(self, *args, **kwargs): return True
    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]: return []
    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        return False

qf8.FineGrainedFP8HfQuantizer = OOTFineGrainedFP8HfQuantizer
# Patch common aliases too (if re-exported)
if hasattr(qz, "FineGrainedFP8HfQuantizer"):
    qz.FineGrainedFP8HfQuantizer = OOTFineGrainedFP8HfQuantizer
# Ensure Auto dispatcher uses our override
qa.AUTO_QUANTIZER_MAPPING["fp8"] = OOTFineGrainedFP8HfQuantizer


from auto_round.utils.model import dequant_block_fp8_weight


class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        block_size: tuple[int, int] | None = (128, 128),
        activation_scheme="dynamic",
    ):
        super().__init__(in_features, out_features)

        # If block size is None, it means that we are doing per-tensor quantization
        self.block_size = block_size
        self.activation_scheme = activation_scheme

        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))

        if self.block_size is None:
            self.weight_scale_inv = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            scale_out_features = (out_features + self.block_size[0] - 1) // self.block_size[0]
            scale_in_features = (in_features + self.block_size[1] - 1) // self.block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )

        if self.activation_scheme == "static":
            self.activation_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def qdq_input(self, bf16_input: torch.Tensor):
        input_scale, input_fp8 = quant_tensor(bf16_input)
        qdq_input_bf16 = input_fp8.to(bf16_input.dtype) * input_scale
        return qdq_input_bf16
    

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dequant_weight = dequant_block_fp8_weight(
            self.weight,
            self.weight_scale_inv,
            block_size=self.block_size,
        )
        dequant_weight = dequant_weight.to(input.dtype)
        # input = self.qdq_input(input)
        out = torch.nn.functional.linear(input, dequant_weight, self.bias)
        return out.to(input.dtype)

torch.nn.Linear = FP8Linear