from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

import transformers.integrations.finegrained_fp8 as finegrained_fp8
from transformers.activations import ACT2FN
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightConverter
from transformers.integrations import use_experts_implementation
from transformers.monkey_patching import get_patch_mapping, register_patch_mapping, unregister_patch_mapping
from transformers.quantizers.quantizers_utils import should_convert_module


_FP8_DTYPE = torch.float8_e4m3fn
_MXFP4_SCALE_DTYPE = torch.float8_e8m0fnu
_PATCH_MAPPING = {
    "DeepseekV4Experts": None,
    "DeepseekV4GroupedLinear": None,
}
_ORIGINAL_LOAD_DEEPGEMM_KERNEL = finegrained_fp8._load_deepgemm_kernel


def _first_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of: {names}")


class MXFP4Linear(nn.Module):
    _FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scale", torch.empty(out_features, in_features // 32, dtype=_MXFP4_SCALE_DTYPE))
        if has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def _unpack_fp4(self, packed: torch.Tensor) -> torch.Tensor:
        lut = torch.tensor(self._FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
        u8 = packed.contiguous().view(torch.uint8)
        low = (u8 & 0xF).long()
        high = ((u8 >> 4) & 0xF).long()
        unpacked = torch.stack([lut[low], lut[high]], dim=-1)
        return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])

    def _dequantize_one(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)

        rows, cols = quantized_fp32.shape[-2:]
        scale_rows, scale_cols = scales.shape[-2:]
        if rows % scale_rows or cols % scale_cols:
            raise ValueError(
                f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols})."
            )

        block_m = rows // scale_rows
        block_n = cols // scale_cols
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        original_shape = quantized_fp32.shape
        q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
        s = scales.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
        return (q * s).to(out_dtype).reshape(original_shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_one(self.weight, self.scale).to(inputs.dtype)
        return F.linear(inputs, weight, self.bias)


class StandaloneFP8Linear(finegrained_fp8.FP8Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        has_bias: bool = False,
        dtype: torch.dtype = _FP8_DTYPE,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            block_size=block_size,
            activation_scheme=activation_scheme,
            has_bias=has_bias,
            dtype=dtype,
        )
        scale = self.weight_scale_inv
        del self._parameters["weight_scale_inv"]
        self.register_parameter("scale", scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.weight_scale_inv = self.scale.to(torch.float32)
        return super().forward(inputs)


class StandaloneDeepseekV4Expert(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = _first_attr(config, "moe_intermediate_size", "intermediate_size")
        self.act_fn = ACT2FN[_first_attr(config, "hidden_activation", "hidden_act")]
        self.limit = config.swiglu_limit
        self.w1 = MXFP4Linear(self.hidden_dim, self.intermediate_dim, has_bias=False)
        self.w3 = MXFP4Linear(self.hidden_dim, self.intermediate_dim, has_bias=False)
        self.w2 = MXFP4Linear(self.intermediate_dim, self.hidden_dim, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.w1(hidden_states).clamp(max=self.limit)
        up = self.w3(hidden_states).clamp(min=-self.limit, max=self.limit)
        return self.w2(self.act_fn(gate) * up)


class StandaloneDeepseekV4Experts(nn.ModuleList):
    def __init__(
        self,
        config,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        has_bias: bool = False,
        has_gate: bool = True,
        dtype: torch.dtype = _FP8_DTYPE,
    ):
        super().__init__()
        if has_bias:
            raise ValueError("StandaloneDeepseekV4Experts does not support bias.")

        self.config = config
        self.has_bias = has_bias
        self.has_gate = has_gate
        self.block_size = block_size
        self.activation_scheme = activation_scheme
        self.dtype = dtype
        self.hidden_dim = config.hidden_size
        self.num_experts = _first_attr(config, "num_local_experts", "num_experts")
        self.intermediate_dim = _first_attr(config, "moe_intermediate_size", "intermediate_size")
        self.extend(StandaloneDeepseekV4Expert(config) for _ in range(self.num_experts))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self[expert_idx](hidden_states[token_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class StandaloneDeepseekV4GroupedLinear(nn.Linear):
    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups
        self.block_size = None
        self.register_parameter("scale", None)

    def enable_fp8_weight_loading(self, block_size: tuple[int, int] | None, pre_quantized: bool = False):
        self.block_size = tuple(block_size) if block_size is not None else None
        if not pre_quantized:
            return

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=_FP8_DTYPE))
        if self.block_size is None:
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            scale_out_features = (self.out_features + self.block_size[0] - 1) // self.block_size[0]
            scale_in_features = (self.in_features + self.block_size[1] - 1) // self.block_size[1]
            self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))

    def _dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.scale is None:
            raise ValueError("StandaloneDeepseekV4GroupedLinear is missing `scale` for FP8 weight loading.")

        if self.block_size is None:
            return (self.weight.float() * self.scale.float()).to(dtype)

        scale_rows, scale_cols = self.scale.shape
        rows, cols = self.weight.shape
        if rows % scale_rows or cols % scale_cols:
            raise ValueError(
                f"Grouped FP8 weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols})."
            )

        block_m = rows // scale_rows
        block_n = cols // scale_cols
        q = self.weight.float().reshape(scale_rows, block_m, scale_cols, block_n)
        s = self.scale.float().unsqueeze(1).unsqueeze(-1)
        return (q * s).reshape(rows, cols).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        if self.weight.element_size() > 1 or self.scale is None:
            weight = self.weight
        else:
            weight = self._dequantize_weight(x.dtype)
        w = weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


_PATCH_MAPPING["DeepseekV4Experts"] = StandaloneDeepseekV4Experts
_PATCH_MAPPING["DeepseekV4GroupedLinear"] = StandaloneDeepseekV4GroupedLinear


def _build_standalone_deepseek_v4_conversion_mapping():
    mapping = get_checkpoint_conversion_mapping("deepseek_v4")
    if mapping is None:
        return None

    filtered = []
    for conversion in mapping:
        target_patterns = getattr(conversion, "_original_target_patterns", None)
        if isinstance(target_patterns, list):
            normalized_targets = set(target_patterns)
        elif target_patterns is None:
            normalized_targets = set()
        else:
            normalized_targets = {target_patterns}

        if isinstance(conversion, WeightConverter) and normalized_targets & {"experts.gate_up_proj", "experts.down_proj"}:
            continue
        filtered.append(conversion)
    return filtered


def _load_deepgemm_kernel():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) not in ((9, 0), (10, 0)):
            raise ImportError(
                f"DeepGEMM requires a Hopper (SM90+) or newer supported GPU recipe, but found {major}.{minor}."
            )
    return _ORIGINAL_LOAD_DEEPGEMM_KERNEL()


def _replace_with_fp8_linear(
    model, modules_to_not_convert: list[str] | None = None, quantization_config=None, pre_quantized: bool = False
):
    if quantization_config.dequantize:
        return model

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        module_kwargs = {} if pre_quantized else {"dtype": None}
        new_module = None
        with torch.device("meta"):
            if module_name.endswith(".experts") and isinstance(module, StandaloneDeepseekV4Experts):
                config = getattr(module, "config", model.config.get_text_config())
                new_module = StandaloneDeepseekV4Experts(
                    config=config,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    has_bias=module.has_bias,
                    has_gate=module.has_gate,
                    **module_kwargs,
                )
            elif module_name.endswith(".experts"):
                has_gate = getattr(module, "has_gate", True)
                has_bias = getattr(module, "has_bias", False)
                config = getattr(module, "config", model.config.get_text_config())
                new_class = use_experts_implementation(
                    experts_class=finegrained_fp8.FP8Experts,
                    experts_interface=finegrained_fp8.ALL_FP8_EXPERTS_FUNCTIONS,
                    has_bias=has_bias,
                    has_gate=has_gate,
                )
                new_module = new_class(
                    config=config,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    has_bias=has_bias,
                    has_gate=has_gate,
                    **module_kwargs,
                )
            elif (
                isinstance(module, nn.Linear)
                and module.__class__.forward is not nn.Linear.forward
                and hasattr(module, "enable_fp8_weight_loading")
            ):
                module.enable_fp8_weight_loading(quantization_config.weight_block_size, pre_quantized=pre_quantized)
                has_been_replaced = True
            elif isinstance(module, nn.Linear) and module.__class__.forward is nn.Linear.forward:
                new_module = StandaloneFP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    has_bias=module.bias is not None,
                    **module_kwargs,
                )

            if new_module is not None:
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        finegrained_fp8.logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )
    return model


def register_deepseek_v4_patches() -> None:
    register_patch_mapping(mapping=_PATCH_MAPPING, overwrite=True)
    finegrained_fp8.replace_with_fp8_linear = _replace_with_fp8_linear
    finegrained_fp8._load_deepgemm_kernel = _load_deepgemm_kernel
    register_checkpoint_conversion_mapping(
        "deepseek_v4",
        _build_standalone_deepseek_v4_conversion_mapping(),
        overwrite=True,
    )


@contextmanager
def deepseek_v4_patch_context():
    previous_mapping = get_patch_mapping()
    previous_replace = finegrained_fp8.replace_with_fp8_linear
    previous_deepgemm_loader = finegrained_fp8._load_deepgemm_kernel
    previous_conversion = get_checkpoint_conversion_mapping("deepseek_v4")

    register_deepseek_v4_patches()
    try:
        yield
    finally:
        finegrained_fp8.replace_with_fp8_linear = previous_replace
        finegrained_fp8._load_deepgemm_kernel = previous_deepgemm_loader
        register_checkpoint_conversion_mapping("deepseek_v4", previous_conversion, overwrite=True)
        for key in _PATCH_MAPPING:
            if key in previous_mapping:
                register_patch_mapping({key: previous_mapping[key]}, overwrite=True)
            else:
                try:
                    unregister_patch_mapping([key])
                except ValueError:
                    pass
