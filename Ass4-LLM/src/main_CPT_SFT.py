import random
import sys
import wandb
import json
from unsloth import FastLanguageModel
import torch
import re
import pandas as pd
import os
from datasets import Dataset
import sys
import gc
import time
sys.stdout.flush()  # 強制刷新輸出緩衝

def preprocess_taigi_text(text):
    """處理台文資料的預處理"""

    # 1. 移除 https/http 開頭的網址，直到遇到標點符號或空格
    # 匹配到 )、。、，、空白 等符號為止
    text = re.sub(r'https?://[^\s)。，！？；：]+', '', text)

    # 2. 移除 www 開頭的網址片段
    text = re.sub(r'www\.[^\s)。，！？；：]+', '', text)

    # 3. 移除殘留的域名片段（更寬鬆的匹配）
    text = re.sub(r'\b\w+\.(com|org|net|edu|gov|tw|io|co|info|biz)(/[^\s)。，！？；：]*)?', '', text)

    # 4. 移除行首的段落編號（如：1. 2. 3.）
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)

    # 5. 統一標點符號
    text = text.replace('。', '。')
    text = text.replace('，', '，')

    # 6. 統一破折號
    text = text.replace('—', '-')

    # 7. 移除過多的空白和換行
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def load_cpt_taigi_data(MAX_TEXT_LENGTH):
    # 收集所有 JSON 檔案的資料
    all_cpt_texts = []

    for file_dir in os.listdir("../data/IMA-Taiwan"):
        dir_path = f"../data/IMA-Taiwan/{file_dir}"

        for file in os.listdir(dir_path):
            if file.endswith(".json"):
                file_path = os.path.join(dir_path, file)
                print(f"讀取: {file_dir}/{file}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

                        # 如果是 list，需要先合併同一篇文章
                        if isinstance(json_data, list):
                            if json_data and 'title' in json_data[0]:
                                from collections import defaultdict
                                # 建立字典，key = title, value = 該文章的所有段落
                                articles = defaultdict(list)

                                for item in json_data:
                                    if 'text' in item and 'title' in item:
                                        title = item['title']
                                        articles[title].append(item['text'])

                                # 處理每一篇文章（每個 title）
                                for title, paragraphs in articles.items():
                                    # 合併同一篇文章的所有段落
                                    full_text = ''.join(paragraphs)

                                    # 一次性處理完整文章
                                    cleaned_text = preprocess_taigi_text(full_text)

                                    # ===== 處理過長的文本 =====
                                    if len(cleaned_text) >= 100:
                                        # 如果文章太長，分段處理
                                        if len(cleaned_text) > MAX_TEXT_LENGTH:
                                            # 按句號切分
                                            sentences = re.split(r'[。！？\n]+', cleaned_text)

                                            current_chunk = ""
                                            for sentence in sentences:
                                                sentence = sentence.strip()
                                                if not sentence:
                                                    continue

                                                # 如果加入這句話會超過限制，先保存當前 chunk
                                                if len(current_chunk) + len(sentence) > MAX_TEXT_LENGTH:
                                                    if len(current_chunk) >= 100:
                                                        all_cpt_texts.append({"text": current_chunk})
                                                    current_chunk = sentence + "。"
                                                else:
                                                    current_chunk += sentence + "。"

                                            # 保存最後一個 chunk
                                            if len(current_chunk) >= 100:
                                                all_cpt_texts.append({"text": current_chunk})
                                        else:
                                            # 文章長度適中，直接加入
                                            all_cpt_texts.append({"text": cleaned_text})

                            # 如果沒有 title，每個元素獨立處理
                            else:
                                for item in json_data:
                                    if 'text' in item:
                                        cleaned_text = preprocess_taigi_text(item['text'])
                                        if 50 <= len(cleaned_text) <= MAX_TEXT_LENGTH:
                                            all_cpt_texts.append({"text": cleaned_text})

                        # 如果是 dict
                        elif isinstance(json_data, dict):
                            if 'text' in json_data:
                                cleaned_text = preprocess_taigi_text(json_data['text'])
                                if 50 <= len(cleaned_text) <= MAX_TEXT_LENGTH:
                                    all_cpt_texts.append({"text": cleaned_text})

                except Exception as e:
                    print(f"讀取 {file_path} 時發生錯誤: {e}")

    print(f"總共讀取了 {len(all_cpt_texts)} 筆 CPT 資料")

    cpt_dataset = Dataset.from_list(all_cpt_texts)

    # 去重
    unique_texts = []
    seen = set()
    for item in all_cpt_texts:
        text = item['text']
        if text not in seen and len(text) > 50:  # 過濾太短的文本
            seen.add(text)
            unique_texts.append(item)
    
    del all_cpt_texts  # 立即釋放
    del seen
    gc.collect()

    print(f"去重後: {len(unique_texts)} 筆")

    # # 查看範例
    # if unique_texts:
    #     print("\n範例文本（前3筆）:")
    #     for i, item in enumerate(unique_texts[:3]):
    #         print(f"\n第 {i+1} 筆 (長度: {len(item['text'])}):")
    #         print(item['text'][:200] + "..." if len(item['text']) > 200 else item['text'])

    cpt_dataset = Dataset.from_list(unique_texts)
    print(f"\n最終訓練資料筆數: {len(cpt_dataset)}")
    return cpt_dataset

def inference(model_path, test_data, output_dir, max_seq_length, batch_size=4):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # 將模型設定為推理模式
    FastLanguageModel.for_inference(model)

    # 設定 tokenizer padding
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 如果輸出檔案已存在，先刪除以避免重複寫入
    if os.path.exists(output_dir):
        os.remove(output_dir)

    test_df = pd.read_csv(test_data)
    # 初始化 write_header 變數
    write_header = True

    print(f"開始預測，總筆數: {len(test_df)}，Batch Size: {batch_size}")

    # 使用 range 每次跳 batch_size 的步長
    for i in range(0, len(test_df), batch_size):
        # 取出目前的 batch 資料
        batch_df = test_df.iloc[i : i + batch_size]

        prompts = []
        ids = []

        # 準備這個 batch 的所有 Prompt
        for index, raw in batch_df.iterrows():
            question_background = raw['前文']
            question = raw['題幹']
            answer1 = raw['選項1']
            answer2 = raw['選項2']
            answer3 = raw['選項3']
            answer4 = raw['選項4']

            # 使用與訓練時相同的格式
            prompt = f"你是一個專業的問答助手，請根據前文的背景，回答題目問題，只要選出正確的選項編號(1-4)。\n前文：{question_background}\n問題：{question}\n從以下四個選項選出正確的選項編號\n選項1：{answer1}\n選項2：{answer2}\n選項3：{answer3}\n選項4：{answer4}\n"

            prompts.append(prompt)
            ids.append(raw['ID'])

        # 批次 tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")

        # 批次生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # 因為只需要生成數字，所以設小一點
                do_sample=False,    # 使用 greedy decoding 確保結果一致
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # 解碼生成的文字（只取新生成的部分）
        predicted_texts = []
        for j, output in enumerate(outputs):
            input_length = inputs['input_ids'][j].shape[0]
            generated_tokens = output[input_length:]
            predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # 移除所有特殊標記和多餘文字
            # 尋找第一個數字 1-4
            match = re.search(r'^[1-4]', predicted_text)
            if match:
                clean_answer = match.group()
            else:
                # 如果沒找到，嘗試從整段文字中找
                match = re.search(r'[1-4]', predicted_text)
                clean_answer = match.group() if match else "1"  # 預設為1

            predicted_texts.append(clean_answer)

        # 建立 Batch 的 DataFrame
        output_batch = pd.DataFrame({
            'ID': ids,
            'Answer': predicted_texts
        })

        # 寫入 CSV (append 模式)
        output_batch.to_csv(output_dir, mode='a', header=write_header, index=False, encoding='utf-8-sig')

        # 第一次寫入後，之後都不需要 header
        write_header = False

        print(f"已處理: {min(i + batch_size, len(test_df))} / {len(test_df)}")

    print("預測完成！")

if __name__ == "__main__":
    #定義模型名稱與最大序列長度
    model_name = "Bohanlu/Taigi-Llama-2-7B"
    wandb_project_name = "Ass4-LLM"
    wandb_name1 = "Taigi-CPT-7B-v2"
    wandb_name2 = "Taigi-SFT-7B-v1"
    max_seq_length = 512
    random_seed = 9527

    print("test")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("❌ CUDA is not available. Using CPU for training.")
        sys.exit(1)
    
    # 登入 wandb
    wandb.login(key="6505e7e06b7f53ea56b61b94658f226c523ebacc")
    os.environ["WANDB_MODE"] = "offline"  # 避免網路問題
    # Start a new wandb run to track this script.
    cpt_run = wandb.init(
        entity="paohuah-national-yang-ming-chiao-tung-university",
        project=wandb_project_name,
        name=wandb_name1,
        config={
            "stage": "CPT",
            "model_name": model_name,
            "learning_rate": 2e-4,
            "max_steps": 1000,
            "lora_r": 64,
            "lora_alpha": 128,
            "max_seq_length": max_seq_length,
            "batch_size": 1,  # 8GB 配置
            "gradient_accumulation": 16,
            "vram": "8GB",
        },
        tags=["CPT", "domain_adaptation"]
    )
    # 載入並處理 CPT 台文資料
    cpt_dataset = load_cpt_taigi_data(max_seq_length)

    # ========== 階段 1: CPT - 持續預訓練 ==========

    # 載入基礎模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # CPT 階段的 LoRA 配置（較大的 rank）
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,  # CPT 用較大的 rank 學習更多知識
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 64,  # 對應調整 alpha
        lora_dropout = 0.05,  # CPT 用較小的 dropout
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = random_seed,
    )

    # CPT 訓練配置
    from trl import SFTTrainer
    from transformers import TrainingArguments

    cpt_trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = cpt_dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = None,  # Windows 必須設為 None
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 16,
            warmup_steps = 100,
            max_steps = 1000,  # CPT 需要更多步驟
            learning_rate = 2e-4,  # CPT 用較高學習率
            fp16 = False,
            bf16 = True,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = random_seed,
            output_dir = "outputs_cpt",
            report_to = "wandb",
            run_name = "Taigi-CPT",
            # 額外優化
            max_grad_norm = 1.0,
            dataloader_num_workers = 0,   # Windows 上設為 0
            dataloader_pin_memory = False,  # 減少 RAM 使用
            save_total_limit = 1,  # 只保留最後一個 checkpoint
        ),
    )

    # 執行 CPT 訓練
    print("開始 CPT 階段訓練...")
    cpt_trainer.train()

    # 儲存 CPT 模型
    model.save_pretrained("../model/cpt_model")
    tokenizer.save_pretrained("../model/cpt_model")
    print("CPT 階段完成！")

    cpt_run.finish()

    # ========== 階段 2: SFT - 監督式微調 ==========
    # 清理記憶體
    del cpt_trainer
    del model
    del tokenizer
    del cpt_dataset
    gc.collect()
    torch.cuda.empty_cache()
    # 等待 3 秒讓系統釋放記憶體
    time.sleep(3)

    sft_run = wandb.init(
        entity="paohuah-national-yang-ming-chiao-tung-university",
        project=wandb_project_name,
        name= wandb_name2,
        config={
            "stage": "SFT",
            "base_model": "./model/cpt_model",
            "learning_rate": 1e-5,
            "max_steps": 100,
            "lora_r": 16,
            "lora_alpha": 32,
            "max_seq_length": max_seq_length,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "vram": "8GB",
        },
        tags=["SFT", "qa_task", "8GB-VRAM"]
    )
    # 載入 CPT 後的模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "../model/cpt_model",
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # 先 merge LoRA weights 到 base model
    model = model.merge_and_unload()

    # 現在可以重新添加新的 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0.1,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = random_seed,
    )

    # 準備 SFT 資料（有標註的問答對）
    df = pd.read_csv("../data/AI_conv.csv")

    sft_dataset_data = []
    for _, row in df.iterrows():
        # 結構化的問答格式
        prompt = f"根據前文內容回答問題\n前文：{row['文章']}\n問題：{row['問題']}\n根據問題，從以下四個選項選出正確的選項編號(1-4)\n選項1：{row['選項1']}\n選項2：{row['選項2']}\n選項3：{row['選項3']}\n選項4：{row['選項4']}\n答案：{str(row['正確答案'])}"

        sft_dataset_data.append({
            "text": prompt
        })

    sft_dataset = Dataset.from_list(sft_dataset_data)

    # SFT 訓練配置
    sft_trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = sft_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = None,  # Windows 必須設為 None
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 16,
            warmup_steps = 10,
            max_steps = 100,  # SFT 步驟較少
            learning_rate = 1e-5,  # SFT 用較小學習率
            fp16 = False,
            bf16 = True,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = random_seed,
            output_dir = "outputs_sft",
            report_to = "wandb",
            run_name = "Taigi-SFT",
            dataloader_num_workers = 0,
            dataloader_pin_memory = False,
        ),
    )

    # 執行 SFT 訓練
    print("開始 SFT 階段訓練...")
    sft_trainer.train()

    # 儲存最終模型
    model.save_pretrained("../model/final_model")
    tokenizer.save_pretrained("../model/final_model")
    print("SFT 階段完成！")

    sft_run.finish()

    # ========== 推理測試 ==========
    inference(
        model_path = "../model/final_model",
        test_data = "../data/1001-question-v3.csv",
        output_dir = "../data/output-CPT&SFT_7B.csv",
        max_seq_length = max_seq_length,
        batch_size = 4
    )