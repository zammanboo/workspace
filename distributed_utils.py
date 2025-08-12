"""
멀티노드 분산 훈련 설정 유틸리티
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import logging

logger = logging.getLogger(__name__)

def setup_distributed(rank, world_size, master_addr="localhost", master_port="12355"):
    """분산 훈련 환경 설정"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    # NCCL 백엔드 초기화
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # 현재 프로세스를 해당 GPU에 바인딩
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    logger.info(f"프로세스 {rank}/{world_size} 초기화 완료 (GPU: {torch.cuda.current_device()})")

def cleanup_distributed():
    """분산 훈련 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    """현재 프로세스의 rank 반환"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """전체 프로세스 수 반환"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process():
    """메인 프로세스인지 확인"""
    return get_rank() == 0

def reduce_tensor(tensor, world_size):
    """텐서를 모든 프로세스에서 평균화"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor

def setup_for_distributed(is_master):
    """분산 환경에서 로깅 설정"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

class DistributedSampler:
    """커스텀 분산 샘플러"""
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 패딩으로 데이터셋 크기를 조정
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # 각 프로세스별로 데이터 분할
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, rank):
    """체크포인트 저장 (FSDP 지원)"""
    if is_main_process():
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        
        # FSDP 모델의 상태 저장
        if isinstance(model, FSDP):
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model_state = model.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"체크포인트 저장 완료: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """체크포인트 로드"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 모델 상태 로드
        if isinstance(model, FSDP):
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 옵티마이저 상태 로드
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 스케줄러 상태 로드
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['loss']
    
    return 0, float('inf')