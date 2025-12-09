"""
台語 CPT 資料預處理腳本
功能：讀取原始 JSON 資料 → 清理 → 去重 → 儲存為 parquet 格式
使用方式：python preprocess_cpt_data.py
"""

import json
import os
import re
import gc
from datasets import Dataset
from tqdm import tqdm

def preprocess_taigi_text(text):
    """處理台文資料的預處理"""
    # 1. 移除 https/http 開頭的網址
    text = re.sub(r'https?://[^\s)。，！？；：]+', '', text)
    # 2. 移除 www 開頭的網址片段
    text = re.sub(r'www\.[^\s)。，！？；：]+', '', text)
    # 3. 移除殘留的域名片段
    text = re.sub(r'\b\w+\.(com|org|net|edu|gov|tw|io|co|info|biz)(/[^\s)。，！？；：]*)?', '', text)
    # 4. 移除行首的段落編號
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    # 5. 統一標點符號
    text = text.replace('。', '。').replace('，', '，')
    # 6. 統一破折號
    text = text.replace('—', '-')
    # 7. 移除過多的空白和換行
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_cpt_data(input_dir, output_file, max_seq_length=1024):
    """
    處理 CPT 資料並儲存
    
    Args:
        input_dir: 輸入資料目錄 (IMA-Taiwan)
        output_file: 輸出檔案路徑 (.parquet 或 .jsonl)
        max_seq_length: 最大序列長度
    """
    
    all_texts = []
    seen_hashes = set()
    
    # 檢查目錄是否存在
    if not os.path.exists(input_dir):
        print(f"錯誤: 目錄 {input_dir} 不存在")
        return
    
    # 計算總檔案數
    total_files = sum([len([f for f in os.listdir(os.path.join(input_dir, d)) if f.endswith('.json')]) 
                       for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    print(f"開始處理 {total_files} 個 JSON 檔案...")
    
    # 使用 tqdm 顯示進度
    with tqdm(total=total_files, desc="處理檔案") as pbar:
        for file_dir in os.listdir(input_dir):
            dir_path = os.path.join(input_dir, file_dir)
            if not os.path.isdir(dir_path):
                continue

            for file in os.listdir(dir_path):
                if not file.endswith(".json"):
                    continue
                    
                file_path = os.path.join(dir_path, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        
                        # 處理不同的 JSON 結構
                        texts = process_json_structure(json_data, max_seq_length)
                        
                        # 去重並加入
                        for text in texts:
                            text_hash = hash(text)
                            if text_hash not in seen_hashes and len(text) >= 100:
                                seen_hashes.add(text_hash)
                                all_texts.append({"text": text})
                        
                except Exception as e:
                    print(f"\n讀取 {file_path} 時發生錯誤: {e}")
                
                pbar.update(1)
    
    print(f"\n處理完成！共 {len(all_texts)} 筆有效資料")
    
    # 轉換為 Dataset 並儲存
    print("建立 Dataset...")
    dataset = Dataset.from_list(all_texts)
    
    # 儲存為 parquet 格式（效率最高）
    if output_file.endswith('.parquet'):
        print(f"儲存為 Parquet 格式: {output_file}")
        dataset.to_parquet(output_file)
    # 或儲存為 JSONL 格式（相容性最好）
    elif output_file.endswith('.jsonl'):
        print(f"儲存為 JSONL 格式: {output_file}")
        dataset.to_json(output_file)
    else:
        print("錯誤: 輸出格式必須是 .parquet 或 .jsonl")
        return
    
    print(f"✅ 資料已儲存至: {output_file}")
    print(f"📊 資料統計:")
    print(f"  - 總筆數: {len(dataset)}")
    print(f"  - 檔案大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # 顯示前 3 筆範例
    print("\n前 3 筆資料範例:")
    for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
        print(f"\n[{i+1}] 長度: {len(example['text'])} 字")
        print(f"內容預覽: {example['text'][:150]}...")
    
    del all_texts, seen_hashes
    gc.collect()


def process_json_structure(json_data, max_seq_length):
    """處理不同結構的 JSON 資料"""
    texts = []
    
    if isinstance(json_data, list):
        # 處理有 title 的文章列表
        if json_data and 'title' in json_data[0]:
            from collections import defaultdict
            articles = defaultdict(list)
            
            for item in json_data:
                if 'text' in item and 'title' in item:
                    articles[item['title']].append(item['text'])
            
            for title, paragraphs in articles.items():
                full_text = ''.join(paragraphs)
                cleaned_text = preprocess_taigi_text(full_text)
                
                if len(cleaned_text) >= 100:
                    # 切分過長文本
                    if len(cleaned_text) > max_seq_length:
                        chunks = split_long_text(cleaned_text, max_seq_length)
                        texts.extend(chunks)
                    else:
                        texts.append(cleaned_text)
        
        # 處理一般列表
        else:
            for item in json_data:
                if 'text' in item:
                    cleaned_text = preprocess_taigi_text(item['text'])
                    if 50 <= len(cleaned_text) <= max_seq_length:
                        texts.append(cleaned_text)
    
    elif isinstance(json_data, dict):
        if 'text' in json_data:
            cleaned_text = preprocess_taigi_text(json_data['text'])
            if 50 <= len(cleaned_text) <= max_seq_length:
                texts.append(cleaned_text)
    
    return texts


def split_long_text(text, max_length):
    """切分過長的文本"""
    chunks = []
    sentences = re.split(r'[。！？\n]+', text)
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) > max_length:
            if len(current_chunk) >= 100:
                chunks.append(current_chunk)
            current_chunk = sentence + "。"
        else:
            current_chunk += sentence + "。"
    
    if len(current_chunk) >= 100:
        chunks.append(current_chunk)
    
    return chunks


if __name__ == "__main__":
    # 設定路徑
    INPUT_DIR = "../data/IMA-Taiwan"
    OUTPUT_FILE = "../data/cpt_dataset.parquet"  # 或 .jsonl
    MAX_SEQ_LENGTH = 1024
    
    print("=" * 60)
    print("台語 CPT 資料預處理")
    print("=" * 60)
    
    # 執行處理
    process_cpt_data(INPUT_DIR, OUTPUT_FILE, MAX_SEQ_LENGTH)
    
    print("\n✅ 處理完成！")
    print(f"下次訓練時，直接載入: {OUTPUT_FILE}")
