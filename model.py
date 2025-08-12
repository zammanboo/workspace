"""
Llama 70B 모델 구조 정의
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import logging

logger = logging.getLogger(__name__)

class LlamaModelForTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaForCausalLM(config)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False  # 훈련 시 메모리 절약
        )
        return outputs

def create_llama_70b_model():
    """Llama 70B 모델 생성"""
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA (Grouped Query Attention)
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
    )
    
    model = LlamaModelForTraining(config)
    logger.info(f"모델 생성 완료 - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    return model, config

def setup_fsdp_model(model, rank):
    """FSDP로 모델 래핑"""
    # Transformer 레이어별로 자동 래핑
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer},
    )
    
    # FSDP 설정
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=rank,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        backward_prefetch=torch.distributed.fsdp.BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=True,
        sync_module_states=True,
    )
    
    return fsdp_model

def get_model_memory_usage(model):
    """모델 메모리 사용량 계산"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return {
        'param_size_gb': param_size / 1e9,
        'buffer_size_gb': buffer_size / 1e9,
        'total_size_gb': total_size / 1e9
    }

if __name__ == "__main__":
    model, config = create_llama_70b_model()
    memory_info = get_model_memory_usage(model)
    print(f"모델 메모리 사용량: {memory_info}")