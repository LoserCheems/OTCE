import math

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class CheemsOTCEConfig(PretrainedConfig):
    model_type = "cheems"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,

        # 基础配置
        # Basic Configuration
        vocab_size=65536,
        type_vocab_size=None,
        hidden_size=1024,
        intermediate_size=1024*4,
        hidden_bias=False,
        hidden_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=65536, # 2^16
        rope_theta=10000.0,

        # 初始化配置
        # Initialization Configuration
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,
        num_logits_to_keep=None,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,

        # SSM 配置
        # SSM Configuration
        use_mamba_kernels=True,
        mamba_act="silu",
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_expand=1,
        mamba_dt_rank="auto",
        mamba_conv_bias=False,
        mamba_proj_bias=False,
        mamba_inner_layernorms=True,
        mamba_rope=True,

        # Attention 配置
        # Attention Configuration
        num_attention_heads=16,
        num_key_value_heads=8,
        attn_implementation="sdpa",
        sliding_window=None,
        n_ctx=262144,
        attention_dropout=0.0,
        attention_rope=True,
        
        # OTCE 配置
        # OTCE Configuration
        num_observation_layers=12,
        num_thinking_layers=2,
        num_conseption_layers=9,
        num_expression_layers=1,

        # 交叉领域MOE配置
        # Cross-domain MOE Configuration
        expert_type="expansive", # expansive or cohesive or None
        num_experts=4,
        num_experts_per_tok=1,
        **kwargs
    ):

        # 基础配置
        # Basic Configuration
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_bias = hidden_bias
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # 初始化配置
        # Initialization Configuration
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # SSM 配置
        # SSM Configuration
        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_act = mamba_act
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_inner_layernorms = mamba_inner_layernorms
        self.mamba_rope = mamba_rope

        # Attention 配置
        # Attention Configuration
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_implementation = attn_implementation
        self.sliding_window = sliding_window
        self.n_ctx = n_ctx
        self.attention_dropout = attention_dropout
        self.attention_rope = attention_rope

        # OTCE 配置
        # OTCE Configuration
        self.num_observation_layers = num_observation_layers
        self.num_thinking_layers = num_thinking_layers
        self.num_conseption_layers = num_conseption_layers
        self.num_expression_layers = num_expression_layers       

        # 交叉领域MOE配置
        # Cross-domain MOE Configuration
        self.expert_type = expert_type
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        super().__init__(
            attn_implementation = attn_implementation,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_hidden_layers(self):
        return self.num_observation_layers + self.num_thinking_layers + self.num_conseption_layers + self.num_expression_layers
    
    @property
    def layers_block_type(self):
        # observation: mamba
        # thinking: attention
        # conseption: mamba
        # expression: attention
        block_type = []
        for i in range(self.num_hidden_layers):
            if i < self.num_observation_layers:
                block_type.append("mamba")
            elif i < self.num_observation_layers + self.num_thinking_layers:
                block_type.append("attention")
            elif i < self.num_observation_layers + self.num_thinking_layers + self.num_conseption_layers:
                block_type.append("mamba")
            else:
                block_type.append("attention")
        return block_type

    @property
    def shared_expert_intermediate_size(self):
        if self.expert_type == "expansive":
            return self.intermediate_size
        else:
            return self.intermediate_size // 2
    
    @property
    def private_expert_intermediate_size(self):
        return self.intermediate_size // 2
