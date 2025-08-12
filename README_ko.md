# Llama 70B 멀티노드 분산 훈련

이 프로젝트는 2개 노드에서 각각 8개의 GPU를 사용하여 Llama 70B 모델을 훈련하는 PyTorch 기반 코드입니다.

## 필요사항

- Python 3.8+
- PyTorch 2.1.0+
- CUDA 11.8+
- 최소 16개 GPU (2 nodes × 8 GPUs)
- 각 GPU당 최소 80GB VRAM (A100/H100 권장)

## 설치

```bash
pip install -r requirements.txt
```

## 파일 구조

- `data_preprocessing.py`: 위키텍스트 데이터 다운로드 및 전처리
- `model.py`: Llama 70B 모델 구조 정의
- `distributed_utils.py`: 멀티노드 분산 훈련 유틸리티
- `train_llama.py`: 메인 훈련 스크립트
- `train_llama_torchrun.py`: torchrun용 훈련 스크립트 (권장)
- `launch_multinode.sh`: 수동 멀티노드 실행 스크립트
- `launch_torchrun.sh`: torchrun 기반 실행 스크립트 (권장)

## 사용법

### 1. torchrun 사용 (권장)

`launch_torchrun.sh` 스크립트의 `MASTER_ADDR`를 마스터 노드의 실제 IP 주소로 수정하세요.

**마스터 노드에서:**
```bash
chmod +x launch_torchrun.sh
./launch_torchrun.sh 0
```

**워커 노드에서:**
```bash
chmod +x launch_torchrun.sh
./launch_torchrun.sh 1
```

### 2. 수동 실행

`launch_multinode.sh` 스크립트의 `MASTER_ADDR`를 마스터 노드의 실제 IP 주소로 수정하세요.

**마스터 노드에서:**
```bash
chmod +x launch_multinode.sh
./launch_multinode.sh master
```

**워커 노드에서:**
```bash
chmod +x launch_multinode.sh
./launch_multinode.sh worker
```

## 주요 특징

- **FSDP (Fully Sharded Data Parallel)**: 메모리 효율적인 70B 모델 훈련
- **Mixed Precision**: FP16을 사용한 메모리 절약 및 속도 향상
- **Gradient Accumulation**: 효과적인 배치 크기 증가
- **체크포인트 저장/로드**: 훈련 중단 시 재시작 가능
- **위키텍스트 데이터**: 자동 다운로드 및 전처리

## 하이퍼파라미터

기본 설정:
- 배치 크기: 1 (per GPU)
- 그래디언트 누적: 8 스텝
- 학습률: 3e-4
- 최대 시퀀스 길이: 2048
- 에포크: 3

## 메모리 요구사항

- Llama 70B 모델: ~140GB (FP16)
- FSDP로 분산 시: 각 GPU당 ~8-10GB
- 배치 크기 1, 시퀀스 길이 2048: 추가 ~4-6GB
- 총 필요 메모리: GPU당 ~15-20GB

## 네트워크 설정

노드 간 통신을 위해 다음을 확인하세요:
- InfiniBand 또는 고속 이더넷 연결
- 방화벽에서 포트 12355 개방
- NCCL 라이브러리 설치

## 트러블슈팅

1. **NCCL 오류**: `export NCCL_DEBUG=INFO`로 디버그 정보 확인
2. **메모리 부족**: 배치 크기나 시퀀스 길이 감소
3. **네트워크 연결 오류**: 마스터 노드 IP 주소 및 포트 확인

## 모니터링

훈련 진행 상황은 다음 방법으로 모니터링할 수 있습니다:
- 콘솔 로그 출력
- `checkpoints/` 디렉토리의 체크포인트 파일
- GPU 사용률: `nvidia-smi`

## 라이선스

이 코드는 교육 및 연구 목적으로 제공됩니다.