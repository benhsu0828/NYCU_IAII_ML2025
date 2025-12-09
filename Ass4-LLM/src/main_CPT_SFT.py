import torch
import argparse
import random
import sys
import wandb
import json
import re
import pandas as pd
import os
from datasets import Dataset, IterableDataset
import sys
import gc
import time
# CPT 訓練配置

from transformers import TrainingArguments
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

def load_cpt_taigi_data(preprocessed_file="../data/cpt_dataset.parquet"):
    """
    載入預處理好的 CPT 資料
    
    Args:
        preprocessed_file: 預處理好的資料檔案路徑 (.parquet 或 .jsonl)
    
    Returns:
        Dataset: 載入的資料集
    """
    if not os.path.exists(preprocessed_file):
        print(f"⚠️  找不到預處理檔案: {preprocessed_file}")
        print("請先執行: python preprocess_cpt_data.py")
        raise FileNotFoundError(f"請先執行資料預處理腳本產生 {preprocessed_file}")
    
    print(f"📂 載入預處理資料: {preprocessed_file}")
    
    # 根據檔案格式載入
    if preprocessed_file.endswith('.parquet'):
        cpt_dataset = Dataset.from_parquet(preprocessed_file)
    elif preprocessed_file.endswith('.jsonl') or preprocessed_file.endswith('.json'):
        cpt_dataset = Dataset.from_json(preprocessed_file)
    else:
        raise ValueError("不支援的檔案格式，請使用 .parquet 或 .jsonl")
    
    print(f"✅ 載入完成！總筆數: {len(cpt_dataset)}")
    print(f"📊 第一筆資料預覽: {cpt_dataset[0]['text'][:100]}...")
    
    return cpt_dataset

def inference(model_path, test_data, output_dir, max_seq_length, batch_size=2):
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
    parser = argparse.ArgumentParser(description="Taigi LLM Training Pipeline")
    parser.add_argument("--run_cpt", action="store_true", help="是否執行 CPT (Continued Pre-Training)")
    parser.add_argument("--run_sft", action="store_true", help="是否執行 SFT (Supervised Fine-Tuning)")
    parser.add_argument("--run_inf", action="store_true", help="是否執行 Inference (預測)")
    args = parser.parse_args()
    
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA 可用：", torch.cuda.get_device_name(0))
        else:
            print("CUDA 不可用，會改用 CPU 或 4-bit on CPU 模式")
    except Exception as e:
        print("檢查 CUDA 時發生錯誤：", e)

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    '''
    model available: 
    Bohanlu/Taigi-Llama-2-7B, 
    unsloth/Qwen2.5-7B-Instruct, 
    unsloth/Yi-1.5-6B-Chat
    '''
    model_name = "unsloth/Qwen2.5-1.5B-Instruct"
    wandb_project_name = "Ass4-LLM"
    wandb_name1 = "Taigi-CPT-Qwen2.5-v1"
    wandb_name2 = "Taigi-SFT-Qwen2.5-v1"
    max_seq_length = 1024
    random_seed = 9527
    cpt_output_dir = "../model/Qwen2.5_cpt_model"
    final_output_dir = "../model/Qwen2.5_final_model"
    inference_output_dir = "../output_Qwen2.5.csv"
    # 登入 wandb
    wandb.login(key="6505e7e06b7f53ea56b61b94658f226c523ebacc")
    # os.environ["WANDB_MODE"] = "offline"  # 避免網路問題
    # Start a new wandb run to track this script.
    if args.run_cpt:
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
        # 載入並處理 CPT 台文資料（使用預處理好的檔案）
        cpt_dataset = load_cpt_taigi_data("../data/cpt_dataset.parquet")

        # ========== 階段 1: CPT - 持續預訓練 ==========

        # 載入基礎模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id


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

        cpt_config = SFTConfig(
            dataset_text_field="text",  # 根據你的 dataset 欄位名稱改
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=100,
            # 你之前用 max_steps，但新版建議用 num_train_epochs
            #num_train_epochs = 50,            # 視你資料量決定
            max_steps=1000,
            learning_rate=2e-4,
            weight_decay=0.01,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            fp16=False,
            bf16=True,
            max_grad_norm=1.0,
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=1,
            output_dir="outputs_cpt",
            # windows 下 data loader 設定如下，減少可能錯誤
            dataloader_pin_memory = False,
            dataloader_num_workers = 0,
            dataset_num_proc=1,
            seed=random_seed,
        )

        cpt_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=cpt_dataset,
            args=cpt_config,
        )
        # cpt_trainer = SFTTrainer(
        #     model = model,
        #     processing_class = tokenizer,
        #     train_dataset = cpt_dataset,
        #     packing = False,
        #     args = SFTConfig(
        #         per_device_train_batch_size = 1,
        #         gradient_accumulation_steps = 16,
        #         warmup_steps = 100,
        #         max_steps = 1000,  # CPT 需要更多步驟
        #         learning_rate = 2e-4,  # CPT 用較高學習率
        #         fp16 = False,
        #         bf16 = True,
        #         logging_steps = 10,
        #         optim = "adamw_8bit",
        #         weight_decay = 0.01,
        #         lr_scheduler_type = "cosine",
        #         seed = random_seed,
        #         output_dir = "outputs_cpt",
        #         report_to = "wandb",
        #         run_name = "Taigi-CPT",
        #         max_length = max_seq_length,
        #         # 額外優化
        #         max_grad_norm = 1.0,
        #         dataloader_num_workers = 0,   # Windows 上設為 0
        #         dataloader_pin_memory = False,  # 減少 RAM 使用
        #         save_total_limit = 1,  # 只保留最後一個 checkpoint
        #     ),
        # )

        # 執行 CPT 訓練
        print("開始 CPT 階段訓練...")
        cpt_trainer.train()

        # 儲存 CPT 模型
        model.save_pretrained(cpt_output_dir)
        tokenizer.save_pretrained(cpt_output_dir)
        print("CPT 階段完成！")

        cpt_run.finish()
        # 清理記憶體
        del cpt_trainer
        del model
        del tokenizer
        del cpt_dataset
        gc.collect()
        torch.cuda.empty_cache()
        # 等待 3 秒讓系統釋放記憶體
        time.sleep(3)
    else:
        print("跳過 CPT 階段")

    # ========== 階段 2: SFT - 監督式微調 ==========
    if args.run_sft:
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
        if args.run_cpt or os.path.exists(cpt_output_dir):
            load_model_path = cpt_output_dir
            print(f"載入 CPT 模型: {load_model_path}")
        else:
            load_model_path = model_name
            print(f"載入原始模型: {load_model_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = load_model_path,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id


        model = FastLanguageModel.get_peft_model(
            model,
            r = 4,  # 降低到 4 減少記憶體
            target_modules = ["q_proj", "v_proj"],  # 只訓練 2 個模組
            lora_alpha = 8,  # 對應調整
            lora_dropout = 0.1,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = random_seed,
        )

        model.train()

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
        sft_config = SFTConfig(
            dataset_text_field="text",  # 根據你的 dataset 欄位名稱改
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            # 你之前用 max_steps，但新版建議用 num_train_epochs
            #num_train_epochs = 5,            # 視你資料量決定
            max_steps=1000,
            learning_rate=1e-5,
            weight_decay=0.01,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            fp16=False,
            bf16=True,
            max_grad_norm=1.0,
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=1,
            output_dir="outputs_cpt",
            # windows 下 data loader 設定如下，減少可能錯誤
            dataloader_pin_memory = False,
            dataloader_num_workers = 0,
            dataset_num_proc=1,
            seed=random_seed,
        )

        sft_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=sft_dataset,
            args=sft_config,  
        )
        # # SFT 訓練配置
        # sft_trainer = SFTTrainer(
        #     model = model,
        #     processing_class = tokenizer, 
        #     train_dataset = sft_dataset,
        #     packing = False,
        #     args = SFTConfig(
        #         per_device_train_batch_size = 1,
        #         gradient_accumulation_steps = 16,
        #         warmup_steps = 10,
        #         max_steps = 100,  # SFT 步驟較少
        #         learning_rate = 1e-5,  # SFT 用較小學習率
        #         fp16 = False,
        #         bf16 = True,
        #         logging_steps = 1,
        #         optim = "adamw_8bit",
        #         weight_decay = 0.01,
        #         lr_scheduler_type = "cosine",
        #         seed = random_seed,
        #         output_dir = "outputs_sft",
        #         report_to = "wandb",
        #         run_name = "Taigi-SFT",
        #         max_length = max_seq_length,
        #         dataloader_num_workers = 0,
        #         dataloader_pin_memory = False,
        #     ),
        # )

        # 執行 SFT 訓練
        print("開始 SFT 階段訓練...")
        sft_trainer.train()

        # 儲存最終模型
        model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        print("SFT 階段完成！")

        sft_run.finish()
    else:
        print("跳過 SFT 階段")

    # ========== 推理測試 ==========
    if args.run_inf:
         # 決定推理用的模型路徑
        if args.run_sft or os.path.exists(final_output_dir):
            inference_model_path = final_output_dir
            print(f"Inference 使用 SFT 後的模型: {inference_model_path}")
        elif args.run_cpt or os.path.exists(cpt_output_dir):
            inference_model_path = cpt_output_dir
            print(f"Inference 使用 CPT 後的模型: {inference_model_path}")
        else:
            inference_model_path = model_name
            print(f"Inference 使用原始模型: {inference_model_path}")

        inference(
            model_path = inference_model_path,
            test_data = "../data/1001-question-v3.csv",
            output_dir = inference_output_dir,
            max_seq_length = max_seq_length,
            batch_size = 2
        )
    else:
        print("跳過推理測試階段")