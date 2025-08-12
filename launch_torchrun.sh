#!/bin/bash

# torchrun을 사용한 멀티노드 훈련 런처 스크립트 (권장)

# 설정 변수
MASTER_ADDR="192.168.1.100"  # 마스터 노드 IP 주소로 변경 필요
MASTER_PORT="12355"
NNODES=2     # 노드 수
NPROC_PER_NODE=8  # 노드당 프로세스(GPU) 수

# 훈련 하이퍼파라미터
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=3e-4
MAX_EPOCHS=3
MAX_LENGTH=2048

# 경로 설정
CHECKPOINT_DIR="./checkpoints"
MODEL_NAME="meta-llama/Llama-2-70b-hf"

echo "========================================="
echo "Llama 70B torchrun 멀티노드 훈련"
echo "========================================="
echo "마스터 주소: $MASTER_ADDR"
echo "마스터 포트: $MASTER_PORT"
echo "노드 수: $NNODES"
echo "노드당 프로세스 수: $NPROC_PER_NODE"
echo "========================================="

# 현재 노드 rank 확인
if [ "$1" == "0" ]; then
    NODE_RANK=0
    echo "마스터 노드 (rank 0)에서 실행"
elif [ "$1" == "1" ]; then
    NODE_RANK=1
    echo "워커 노드 (rank 1)에서 실행"
else
    echo "사용법: $0 [0|1]"
    echo "예시:"
    echo "  마스터 노드: $0 0"
    echo "  워커 노드: $0 1"
    exit 1
fi

# NCCL 환경 변수 설정
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0

# torchrun으로 실행
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_llama_torchrun.py \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_epochs $MAX_EPOCHS \
    --max_length $MAX_LENGTH \
    --model_name $MODEL_NAME \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_interval 10 \
    --save_interval 500

echo "훈련 완료!"