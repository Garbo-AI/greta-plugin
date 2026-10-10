"""Out-of-tree Greta model registration. No CUDA imports in the plugin hook."""

def register():
    from transformers import AutoConfig
    from greta.configuration_greta import GretaConfig
    from vllm import ModelRegistry
    from vllm.config.compilation import CompilationConfig
    from .tool_parser import GretaToolParser
    from .reasoning_parser import GretaReasoningParser
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    from vllm.reasoning import ReasoningParserManager

    AutoConfig.register('Greta', GretaConfig, exist_ok=True)
    if 'GretaForCausalLM' not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            'GretaForCausalLM', 'greta_vllm.model:GretaForCausalLM'
        )
    # The dynamic convolution splits the compiled graph like vLLM's own
    # short_conv/mamba ops: mixed batches run it eagerly between piecewise
    # CUDA graphs. Must be set before vLLM builds its config (plugins load first).
    if 'vllm::greta_dynamic_conv' not in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops.append('vllm::greta_dynamic_conv')
    ToolParserManager.register_module(["greta", "greta_markers"], module=GretaToolParser)
    ReasoningParserManager.register_module(["greta", "greta_markers"], module=GretaReasoningParser)
