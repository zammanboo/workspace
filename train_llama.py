"""
Llama 70B 멀티노드 분산 훈련 메인 스크립트
"""
import os
import sys
import math
import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType

# 로컬 모듈 임포트
from model import create_llama_70b_model, setup_fsdp_model
from data_preprocessing import prepare_data
from distributed_utils import (
    setup_distributed, cleanup_distributed, get_rank, get_world_size,
    is_main_process, reduce_tensor, save_checkpoint, load_checkpoint,
    setup_for_distributed
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Llama 70B 멀티노드 훈련')
    
    # 분산 훈련 설정
    parser.add_argument('--world_size', type=int, default=16, help='전체 프로세스 수 (2 nodes × 8 GPUs)')
    parser.add_argument('--rank', type=int, default=0, help='현재 프로세스 rank')
    parser.add_argument('--master_addr', type=str, default='localhost', help='마스터 노드 주소')
    parser.add_argument('--master_port', type=str, default='12355', help='마스터 포트')
    
    # 훈련 하이퍼파라미터
    parser.add_argument('--batch_size', type=int, default=1, help='배치 크기 (per GPU)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='그래디언트 누적 스텝')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='가중치 감소')
    parser.add_argument('--max_epochs', type=int, default=3, help='최대 에포크')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='워밍업 스텝')
    parser.add_argument('--max_length', type=int, default=2048, help='최대 시퀀스 길이')
    
    # 모델 및 데이터 경로
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-70b-hf', help='모델 이름')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='체크포인트 저장 경로')
    parser.add_argument('--resume_from', type=str, default=None, help='재시작할 체크포인트 경로')
    
    # 로깅 및 저장
    parser.add_argument('--log_interval', type=int, default=100, help='로그 출력 간격')
    parser.add_argument('--save_interval', type=int, default=1000, help='체크포인트 저장 간격')
    
    return parser.parse_args()

def create_optimizer(model, args):
    """옵티마이저 생성"""
    # 가중치 감소를 적용하지 않을 파라미터들
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    return optimizer

def create_scheduler(optimizer, num_training_steps, args):
    """학습률 스케줄러 생성"""
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler

def train_epoch(model, dataloader, optimizer, scheduler, epoch, args):
    """한 에포크 훈련"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())):
        # 데이터를 GPU로 이동
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        
        # 순전파
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / args.gradient_accumulation_steps
        
        # 역전파
        loss.backward()
        
        # 그래디언트 누적
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 옵티마이저 스텝
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 손실 값 누적
        total_loss += loss.item()
        
        # 로깅
        if step % args.log_interval == 0 and is_main_process():
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch}, Step {step}/{num_batches}, "
                       f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # 체크포인트 저장
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                total_loss / (step + 1), args.checkpoint_dir, get_rank()
            )
    
    # 평균 손실 계산
    avg_loss = total_loss / num_batches
    
    # 분산 환경에서 손실 평균화
    loss_tensor = torch.tensor(avg_loss).cuda()
    avg_loss = reduce_tensor(loss_tensor, get_world_size()).item()
    
    return avg_loss

def main():
    args = parse_args()
    
    # 분산 훈련 설정
    setup_distributed(args.rank, args.world_size, args.master_addr, args.master_port)
    setup_for_distributed(is_main_process())
    
    # 체크포인트 디렉토리 생성
    if is_main_process():
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 모델 생성
    logger.info("모델 생성 중...")
    model, config = create_llama_70b_model()
    
    # FSDP로 모델 래핑
    logger.info("FSDP 모델 설정 중...")
    model = setup_fsdp_model(model, get_rank())
    
    # 데이터 준비
    logger.info("데이터 준비 중...")
    train_dataset, val_dataset, tokenizer = prepare_data(
        tokenizer_path=args.model_name,
        max_length=args.max_length
    )
    
    # 데이터로더 생성
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 옵티마이저 및 스케줄러 생성
    optimizer = create_optimizer(model, args)
    
    num_training_steps = len(train_dataloader) * args.max_epochs // args.gradient_accumulation_steps
    scheduler = create_scheduler(optimizer, num_training_steps, args)
    
    # 체크포인트에서 재시작
    start_epoch = 0
    if args.resume_from:
        start_epoch, _ = load_checkpoint(args.resume_from, model, optimizer, scheduler)
        logger.info(f"에포크 {start_epoch}부터 재시작")
    
    # 훈련 루프
    logger.info("훈련 시작...")
    for epoch in range(start_epoch, args.max_epochs):
        train_sampler.set_epoch(epoch)
        
        avg_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch, args)
        
        if is_main_process():
            logger.info(f"Epoch {epoch} 완료 - 평균 손실: {avg_loss:.4f}")
        
        # 에포크 종료 시 체크포인트 저장
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            avg_loss, args.checkpoint_dir, get_rank()
        )
    
    # 훈련 완료
    logger.info("훈련 완료!")
    cleanup_distributed()

if __name__ == "__main__":
    main()