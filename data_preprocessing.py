"""
위키텍스트 데이터 다운로드 및 전처리
"""
import os
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer
import json
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=2048, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 위키텍스트 데이터셋 로드
        if not os.path.exists(data_path):
            logger.info("위키텍스트 데이터셋 다운로드 중...")
            dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
            
            # 텍스트 전처리 및 토크나이징
            logger.info("데이터 전처리 중...")
            for item in dataset:
                text = item['text'].strip()
                if len(text) > 10:  # 너무 짧은 텍스트 제외
                    # 토크나이징
                    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
                    if len(tokens) > 50:  # 최소 길이 보장
                        self.data.append(tokens)
            
            # 전처리된 데이터 저장
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w') as f:
                json.dump(self.data, f)
            logger.info(f"전처리된 데이터 저장 완료: {data_path}")
        else:
            # 기존 전처리된 데이터 로드
            logger.info(f"기존 전처리된 데이터 로드: {data_path}")
            with open(data_path, 'r') as f:
                self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        
        # 패딩 또는 잘라내기
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # 입력과 타겟 설정 (다음 토큰 예측)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }

def prepare_data(tokenizer_path="meta-llama/Llama-2-70b-hf", max_length=2048):
    """데이터 준비 함수"""
    # 토크나이저 로드
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    
    # 특수 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 데이터셋 생성
    train_dataset = WikiTextDataset(
        tokenizer=tokenizer,
        data_path="./data/wikitext_train.json",
        max_length=max_length,
        split='train'
    )
    
    val_dataset = WikiTextDataset(
        tokenizer=tokenizer,
        data_path="./data/wikitext_val.json",
        max_length=max_length,
        split='validation'
    )
    
    return train_dataset, val_dataset, tokenizer

if __name__ == "__main__":
    train_dataset, val_dataset, tokenizer = prepare_data()
    print(f"훈련 데이터 크기: {len(train_dataset)}")
    print(f"검증 데이터 크기: {len(val_dataset)}")
    print(f"토크나이저 vocabulary 크기: {len(tokenizer)}")