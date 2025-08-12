#!/bin/bash

# Llama 70B 멀티노드 훈련 런처 스크립트
# 2개 노드, 각 노드당 8개 GPU 사용

# 설정 변수
MASTER_ADDR="192.168.1.100"  # 마스터 노드 IP 주소로 변경 필요
MASTER_PORT="12355"
WORLD_SIZE=16  # 2 nodes × 8 GPUs = 16
GPUS_PER_NODE=8

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
echo "Llama 70B 멀티노드 훈련 시작"
echo "========================================="
echo "마스터 주소: $MASTER_ADDR"
echo "마스터 포트: $MASTER_PORT"
echo "전체 프로세스 수: $WORLD_SIZE"
echo "노드당 GPU 수: $GPUS_PER_NODE"
echo "배치 크기: $BATCH_SIZE"
echo "그래디언트 누적 스텝: $GRADIENT_ACCUMULATION_STEPS"
echo "학습률: $LEARNING_RATE"
echo "최대 에포크: $MAX_EPOCHS"
echo "========================================="

# 현재 노드가 마스터 노드인지 확인
if [ "$1" == "master" ]; then
    echo "마스터 노드에서 실행 중..."
    NODE_RANK=0
    FIRST_GPU_RANK=0
elif [ "$1" == "worker" ]; then
    echo "워커 노드에서 실행 중..."
    NODE_RANK=1
    FIRST_GPU_RANK=8
else
    echo "사용법: $0 [master|worker]"
    echo "예시:"
    echo "  마스터 노드: $0 master"
    echo "  워커 노드: $0 worker"
    exit 1
fi

# 환경 변수 설정
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0

# 각 GPU별로 프로세스 실행
for ((i=0; i<$GPUS_PER_NODE; i++)); do
    LOCAL_RANK=$i
    GLOBAL_RANK=$((FIRST_GPU_RANK + i))
    
    echo "GPU $LOCAL_RANK에서 글로벌 rank $GLOBAL_RANK 프로세스 시작..."
    
    CUDA_VISIBLE_DEVICES=$LOCAL_RANK python -u train_llama.py \
        --world_size $WORLD_SIZE \
        --rank $GLOBAL_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate $LEARNING_RATE \
        --max_epochs $MAX_EPOCHS \
        --max_length $MAX_LENGTH \
        --model_name $MODEL_NAME \
        --checkpoint_dir $CHECKPOINT_DIR \
        --log_interval 10 \
        --save_interval 500 &
done

# 모든 백그라운드 프로세스가 완료될 때까지 대기
wait

echo "========================================="
echo "훈련 완료!"
echo "========================================="