# 📝 方案 1: 基於訓練集詞頻的拼寫修正（推薦）

import pandas as pd
from collections import Counter
import re
from Levenshtein import distance as levenshtein_distance

# ==================== 步驟 1: 建立訓練集詞彙表 ====================

print("📚 步驟 1: 建立訓練集詞彙表...")

# 讀取訓練集
train_csv = "./train/train/trainAgg-toneless.csv"
train_df = pd.read_csv(train_csv)

# 提取所有文字
all_texts = ' '.join(train_df['text'].astype(str))

# ✅ 分詞 按空格分割（如果訓練集有分詞）
words = all_texts.split()

# 統計詞頻
word_freq = Counter(words)

print(f"✅ 詞彙表大小: {len(word_freq)}")
print(f"   總詞數: {sum(word_freq.values())}")
print(f"\n前 20 個高頻字:")
for word, freq in word_freq.most_common(20):
    print(f"   '{word}': {freq} 次")

# ==================== 步驟 2: 定義拼寫修正函數 ====================

def get_candidates(word, word_freq, max_distance=2):
    """
    取得候選修正詞（基於編輯距離）
    
    Args:
        word: 待修正的詞
        word_freq: 詞頻字典
        max_distance: 最大編輯距離
    
    Returns:
        候選詞列表 [(候選詞, 編輯距離, 詞頻)]
    """
    candidates = []
    
    for vocab_word, freq in word_freq.items():
        dist = levenshtein_distance(word, vocab_word)
        if dist <= max_distance and dist > 0:  # ✅ 排除原詞
            candidates.append((vocab_word, dist, freq))
    
    # 按優先級排序：編輯距離越小越好，詞頻越高越好
    candidates.sort(key=lambda x: (x[1], -x[2]))
    
    return candidates

def correct_word(word, word_freq, threshold=2):
    """
    修正單個詞
    
    Args:
        word: 待修正的詞
        word_freq: 詞頻字典
        threshold: 編輯距離閾值
    
    Returns:
        修正後的詞
    """
    # 如果詞在詞彙表中，直接返回
    if word in word_freq:
        return word
    
    # 取得候選詞
    candidates = get_candidates(word, word_freq, max_distance=threshold)
    
    # 如果有候選詞，返回最佳候選
    if candidates:
        best_candidate = candidates[0][0]
        return best_candidate
    
    # 沒有候選詞，返回原詞
    return word

def correct_sentence(sentence, word_freq, threshold=2):
    """
    修正整個句子
    
    Args:
        sentence: 待修正的句子
        word_freq: 詞頻字典
        threshold: 編輯距離閾值
    
    Returns:
        修正後的句子
    """
    # ✅ 方法 1: 按字元修正（台語常見）
    words = list(sentence.replace(' ', ''))
    
    # ✅ 方法 2: 按空格修正（如果有分詞）
    # words = sentence.split()
    
    corrected_words = [correct_word(word, word_freq, threshold) for word in words]
    
    # ✅ 方法 1: 不加空格（台語常見）
    return ''.join(corrected_words)
    
    # ✅ 方法 2: 加空格（如果有分詞）
    # return ' '.join(corrected_words)

print("✅ 拼寫修正函數定義完成")

# ==================== 步驟 3: 修正測試集預測結果 ====================

print("\n📝 步驟 3: 修正測試集預測結果...")

# 讀取預測結果
submission_csv = "submission_JacobLinCool.csv"
predictions_df = pd.read_csv(submission_csv)

print(f"📊 原始預測筆數: {len(predictions_df)}")

# 修正每個預測
corrected_sentences = []
correction_count = 0

for idx, row in predictions_df.iterrows():
    original_sentence = row['sentence']
    corrected_sentence = correct_sentence(original_sentence, word_freq, threshold=2)
    
    corrected_sentences.append(corrected_sentence)
    
    # 統計修正次數
    if original_sentence != corrected_sentence:
        correction_count += 1
        if correction_count <= 10:  # 顯示前 10 個修正範例
            print(f"\n修正範例 {correction_count}:")
            print(f"   原始: {original_sentence}")
            print(f"   修正: {corrected_sentence}")

# 更新 DataFrame
predictions_df['sentence'] = corrected_sentences

# ==================== 步驟 4: 儲存修正後的結果 ====================

corrected_csv = submission_csv + "_corrected.csv"
predictions_df.to_csv(corrected_csv, index=False)

print(f"\n{'='*60}")
print(f"✅ 拼寫修正完成！")
print(f"{'='*60}")
print(f"📊 統計:")
print(f"   總筆數: {len(predictions_df)}")
print(f"   修正筆數: {correction_count}")
print(f"   修正比例: {correction_count / len(predictions_df) * 100:.2f}%")
print(f"\n💾 輸出檔案:")
print(f"   原始: {submission_csv}")
print(f"   修正: {corrected_csv}")
print(f"{'='*60}")