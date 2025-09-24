#!/usr/bin/env python3
"""
數據預處理 V2 - 包含特徵關係分析和可自定義的數據處理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
import sys
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 加入 src 目錄到路徑
sys.path.append(str(Path(__file__).parent))
from data_preprocess import load_data

class DataAnalysisV2:
    """數據分析和預處理 V2"""
    
    def __init__(self):
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.analysis_results = {}
        
    def load_data(self):
        """載入原始數據"""
        print("=== 載入原始數據 ===")
        self.train_df, self.valid_df, self.test_df = load_data()
        print(f"✅ 數據載入完成!")
        print(f"   訓練集: {self.train_df.shape}")
        print(f"   驗證集: {self.valid_df.shape}")
        print(f"   測試集: {self.test_df.shape}")
        
    def analyze_data_relationship(self, target_column='總價元', save_results=True):
        """分析數據特徵與目標變數的關係"""
        if self.train_df is None:
            print("❌ 請先載入數據!")
            return None
            
        print("\n=== 開始特徵關係分析 ===")
        
        # 1. 基本統計
        print("\n1. 目標變數基本統計:")
        target_stats = {
            'count': self.train_df[target_column].count(),
            'mean': self.train_df[target_column].mean(),
            'median': self.train_df[target_column].median(),
            'std': self.train_df[target_column].std(),
            'min': self.train_df[target_column].min(),
            'max': self.train_df[target_column].max(),
            'skewness': self.train_df[target_column].skew()
        }
        
        for key, value in target_stats.items():
            print(f"   {key}: {value:,.2f}")
        
        # 2. 特徵類型分析
        print(f"\n2. 特徵類型分析:")
        numeric_features = []
        categorical_features = []
        
        for col in self.train_df.columns:
            if col == target_column:
                continue
                
            if self.train_df[col].dtype in ['int64', 'float64']:
                # 檢查是否為類別型數值 (唯一值很少)
                unique_vals = self.train_df[col].nunique()
                if unique_vals <= 20:  # 可能是類別型
                    print(f"   🔍 {col}: 數值型但唯一值少 ({unique_vals}個), 可能是類別型")
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        print(f"   📊 數值型特徵: {len(numeric_features)}")
        print(f"   📋 類別型特徵: {len(categorical_features)}")
        
        # 3. 相關性分析
        print(f"\n3. 相關性分析:")
        correlations = self._calculate_correlations(numeric_features, target_column)
        
        # 4. 互信息分析
        print(f"\n4. 互信息分析:")
        mi_scores = self._calculate_mutual_information(numeric_features, target_column)
        
        # 5. 綜合排序
        print(f"\n5. 綜合特徵重要性:")
        ranked_features = self._rank_features(correlations, mi_scores)
        
        # 6. 缺失值分析
        print(f"\n6. 缺失值分析:")
        missing_analysis = self._analyze_missing_values()
        
        # 儲存結果
        self.analysis_results = {
            'target_stats': target_stats,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'correlations': correlations,
            'mutual_info': mi_scores,
            'ranked_features': ranked_features,
            'missing_analysis': missing_analysis
        }
        
        # 生成報告
        if save_results:
            self._generate_analysis_report()
            self._plot_feature_importance()
        
        return self.analysis_results
    
    def _calculate_correlations(self, numeric_features, target_column):
        """計算相關性"""
        correlations = {}
        
        print(f"   📈 計算皮爾森相關係數...")
        for feature in numeric_features:
            try:
                # 移除缺失值
                mask = ~(self.train_df[feature].isna() | self.train_df[target_column].isna())
                if mask.sum() < 10:  # 樣本太少
                    continue
                    
                x = self.train_df.loc[mask, feature]
                y = self.train_df.loc[mask, target_column]
                
                pearson_r, pearson_p = pearsonr(x, y)
                spearman_r, spearman_p = spearmanr(x, y)
                
                correlations[feature] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'abs_pearson': abs(pearson_r)
                }
                
            except Exception as e:
                print(f"      計算 {feature} 相關性時出錯: {e}")
        
        # 按絕對值排序
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1]['abs_pearson'], reverse=True)
        
        print(f"   🔍 相關性 Top 10:")
        for i, (feature, stats) in enumerate(sorted_corr[:10]):
            significance = "***" if stats['pearson_p'] < 0.001 else "**" if stats['pearson_p'] < 0.01 else "*" if stats['pearson_p'] < 0.05 else ""
            print(f"      {i+1:2d}. {feature[:30]:30s}: {stats['pearson_r']:7.4f} {significance}")
        
        return correlations
    
    def _calculate_mutual_information(self, numeric_features, target_column):
        """計算互信息"""
        mi_scores = {}
        
        try:
            print(f"   🧠 計算互信息分數...")
            
            # 準備數據
            feature_data = []
            feature_names = []
            
            for feature in numeric_features:
                if feature in self.train_df.columns:
                    # 填充缺失值
                    values = self.train_df[feature].fillna(self.train_df[feature].median())
                    feature_data.append(values)
                    feature_names.append(feature)
            
            if not feature_data:
                print(f"      ❌ 沒有可用的數值特徵")
                return mi_scores
            
            X = np.column_stack(feature_data)
            y = self.train_df[target_column]
            
            # 計算互信息
            mi_values = mutual_info_regression(X, y, random_state=42)
            
            for feature, score in zip(feature_names, mi_values):
                mi_scores[feature] = score
            
            # 按分數排序
            sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"   🔍 互信息 Top 10:")
            for i, (feature, score) in enumerate(sorted_mi[:10]):
                print(f"      {i+1:2d}. {feature[:30]:30s}: {score:.6f}")
                
        except Exception as e:
            print(f"   ❌ 互信息計算出錯: {e}")
        
        return mi_scores
    
    def _rank_features(self, correlations, mi_scores):
        """綜合排序特徵"""
        combined_scores = {}
        
        # 正規化互信息分數
        max_mi = max(mi_scores.values()) if mi_scores else 1
        
        for feature in correlations.keys():
            pearson_score = correlations[feature]['abs_pearson']
            mi_score = mi_scores.get(feature, 0) / max_mi
            
            # 綜合分數 (60% 相關性 + 40% 互信息)
            combined_score = 0.6 * pearson_score + 0.4 * mi_score
            
            combined_scores[feature] = {
                'combined_score': combined_score,
                'pearson_r': correlations[feature]['pearson_r'],
                'pearson_abs': pearson_score,
                'mutual_info': mi_scores.get(feature, 0),
                'p_value': correlations[feature]['pearson_p']
            }
        
        # 排序
        ranked = sorted(combined_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        print(f"   🏆 綜合重要性 Top 15:")
        print(f"   {'排名':<4} {'特徵名稱':<30} {'綜合分數':<10} {'相關係數':<10} {'互信息':<10}")
        print(f"   {'-'*70}")
        
        for i, (feature, scores) in enumerate(ranked[:15]):
            print(f"   {i+1:<4} {feature[:30]:<30} {scores['combined_score']:<10.4f} "
                  f"{scores['pearson_r']:<10.4f} {scores['mutual_info']:<10.6f}")
        
        return ranked
    
    def _analyze_missing_values(self):
        """分析缺失值"""
        missing_info = {}
        
        print(f"   🔍 缺失值統計:")
        total_samples = len(self.train_df)
        
        for col in self.train_df.columns:
            missing_count = self.train_df[col].isna().sum()
            missing_pct = (missing_count / total_samples) * 100
            
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
        
        if missing_info:
            # 按缺失比例排序
            sorted_missing = sorted(missing_info.items(), key=lambda x: x[1]['percentage'], reverse=True)
            
            print(f"      發現 {len(missing_info)} 個特徵有缺失值:")
            for feature, info in sorted_missing[:10]:  # 只顯示前10個
                print(f"      • {feature[:30]:30s}: {info['count']:4d} ({info['percentage']:5.1f}%)")
        else:
            print(f"      ✅ 沒有發現缺失值")
        
        return missing_info
    
    def _generate_analysis_report(self):
        """生成分析報告"""
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        # 建立結果目錄
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("🏠 房地產數據特徵關係分析報告 V2")
        report_lines.append("=" * 60)
        report_lines.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 基本統計
        target_stats = self.analysis_results['target_stats']
        report_lines.append(f"\n📊 目標變數統計 (總價元):")
        report_lines.append(f"  樣本數: {target_stats['count']:,}")
        report_lines.append(f"  平均值: {target_stats['mean']:,.0f}")
        report_lines.append(f"  中位數: {target_stats['median']:,.0f}")
        report_lines.append(f"  標準差: {target_stats['std']:,.0f}")
        report_lines.append(f"  範圍: {target_stats['min']:,.0f} ~ {target_stats['max']:,.0f}")
        
        # 特徵統計
        report_lines.append(f"\n📈 特徵統計:")
        report_lines.append(f"  數值型特徵: {len(self.analysis_results['numeric_features'])}")
        report_lines.append(f"  類別型特徵: {len(self.analysis_results['categorical_features'])}")
        
        # 最重要特徵
        report_lines.append(f"\n🏆 最重要的15個特徵:")
        for i, (feature, scores) in enumerate(self.analysis_results['ranked_features'][:15]):
            report_lines.append(f"  {i+1:2d}. {feature:<30} (綜合分數: {scores['combined_score']:.4f})")
        
        # 高相關特徵
        report_lines.append(f"\n🔗 高度相關特徵 (|r| > 0.3):")
        high_corr = []
        for feature, stats in self.analysis_results['correlations'].items():
            if abs(stats['pearson_r']) > 0.3:
                high_corr.append((feature, stats['pearson_r']))
        
        if high_corr:
            high_corr.sort(key=lambda x: abs(x[1]), reverse=True)
            for feature, corr in high_corr:
                direction = "正相關" if corr > 0 else "負相關"
                report_lines.append(f"  • {feature:<30}: {corr:7.4f} ({direction})")
        else:
            report_lines.append("  沒有發現高度相關特徵")
        
        # 缺失值分析
        missing_info = self.analysis_results['missing_analysis']
        if missing_info:
            report_lines.append(f"\n⚠️ 缺失值分析:")
            sorted_missing = sorted(missing_info.items(), key=lambda x: x[1]['percentage'], reverse=True)
            for feature, info in sorted_missing[:10]:
                report_lines.append(f"  • {feature:<30}: {info['percentage']:5.1f}% ({info['count']} 筆)")
        
        # 儲存報告
        report_text = "\n".join(report_lines)
        report_file = results_dir / f"data_analysis_report_v2_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n📄 分析報告已儲存: {report_file}")
        
        # 也打印到控制台
        print(f"\n{report_text}")
    
    def _plot_feature_importance(self):
        """繪製特徵重要性圖表"""
        try:
            ranked_features = self.analysis_results['ranked_features']
            
            # 取前15個特徵
            top_15 = ranked_features[:15]
            features = [item[0][:25] for item, _ in enumerate(top_15)]  # 截短特徵名
            scores = [item[1]['combined_score'] for item in top_15]
            correlations = [item[1]['pearson_r'] for item in top_15]
            
            # 創建圖表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # 圖1: 綜合重要性分數
            y_pos = np.arange(len(features))
            bars1 = ax1.barh(y_pos, scores, color='skyblue', alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([item[0][:25] for item in top_15])
            ax1.set_xlabel('綜合重要性分數')
            ax1.set_title('特徵重要性排序 (綜合分數)')
            ax1.grid(axis='x', alpha=0.3)
            
            # 在條形上顯示數值
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            # 圖2: 相關係數
            colors = ['red' if x < 0 else 'green' for x in correlations]
            bars2 = ax2.barh(y_pos, correlations, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([item[0][:25] for item in top_15])
            ax2.set_xlabel('皮爾森相關係數')
            ax2.set_title('特徵與售價相關性')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(axis='x', alpha=0.3)
            
            # 在條形上顯示數值
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width + (0.01 if width >= 0 else -0.01), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', 
                        ha='left' if width >= 0 else 'right', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # 儲存圖表
            timestamp = datetime.now().strftime("%m%d_%H%M")
            plot_file = Path("../results") / f"feature_importance_v2_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            print(f"📊 特徵重要性圖表已儲存: {plot_file}")
            plt.show()
            
        except Exception as e:
            print(f"❌ 繪圖出錯: {e}")
    
    def process_data_custom(self):
        """自定義數據處理 - 可以根據分析結果進行特徵工程"""
        if self.train_df is None:
            print("❌ 請先載入數據!")
            return None
            
        print("\n=== 開始自定義數據處理 ===")
        
        # 複製數據防止修改原始數據
        train_processed = self.train_df.copy()
        valid_processed = self.valid_df.copy()
        test_processed = self.test_df.copy()
        
        print("📝 以下是需要你自行修改的數據處理部分:")
        print("-" * 50)
        
        # ============ 在這裡添加你的數據處理邏輯 ============
        
        # 1. 處理缺失值
        print("1. 處理缺失值:")
        print("   # TODO: 根據分析結果處理缺失值")
        print("   # 範例:")
        print("   # train_processed['某欄位'].fillna(train_processed['某欄位'].median(), inplace=True)")
        
        # 2. 特徵工程 - 根據分析結果創建新特徵
        print("\n2. 特徵工程:")
        print("   # TODO: 根據相關性分析結果創建新特徵")
        print("   # 範例:")
        print("   # train_processed['新特徵'] = train_processed['特徵1'] * train_processed['特徵2']")
        
        # 3. 類別型特徵編碼
        print("\n3. 類別型特徵編碼:")
        print("   # TODO: 對類別型特徵進行編碼")
        categorical_features = self.analysis_results.get('categorical_features', [])
        if categorical_features:
            print(f"   # 發現的類別型特徵: {categorical_features[:5]}...")
            print("   # 範例:")
            print("   # le = LabelEncoder()")
            print("   # train_processed['某類別欄位'] = le.fit_transform(train_processed['某類別欄位'])")
        
        # 4. 數值特徵處理
        print("\n4. 數值特徵處理:")
        if hasattr(self, 'analysis_results') and 'ranked_features' in self.analysis_results:
            top_features = [item[0] for item in self.analysis_results['ranked_features'][:10]]
            print(f"   # 重要數值特徵 (前10個): {top_features}")
            print("   # TODO: 對重要特徵進行變換或正規化")
            print("   # 範例:")
            print("   # train_processed['重要特徵_log'] = np.log1p(train_processed['重要特徵'])")
        
        # 5. 異常值處理
        print("\n5. 異常值處理:")
        print("   # TODO: 根據需要處理異常值")
        print("   # 範例:")
        print("   # Q1 = train_processed['某欄位'].quantile(0.25)")
        print("   # Q3 = train_processed['某欄位'].quantile(0.75)")
        print("   # IQR = Q3 - Q1")
        print("   # train_processed = train_processed[~((train_processed['某欄位'] < (Q1 - 1.5 * IQR)) | (train_processed['某欄位'] > (Q3 + 1.5 * IQR)))]")
        
        # 6. 特徵選擇
        print("\n6. 特徵選擇:")
        print("   # TODO: 根據分析結果選擇重要特徵")
        print("   # 範例:")
        print("   # selected_features = ['重要特徵1', '重要特徵2', ...]")
        print("   # train_processed = train_processed[selected_features + ['總價元']]")
        
        print("\n" + "=" * 50)
        print("💡 修改建議:")
        if hasattr(self, 'analysis_results'):
            if self.analysis_results.get('missing_analysis'):
                print("• 優先處理缺失值較多的特徵")
            
            ranked_features = self.analysis_results.get('ranked_features', [])
            if ranked_features:
                print(f"• 重點關注前10個重要特徵: {[item[0] for item in ranked_features[:10]]}")
            
            high_corr_features = []
            correlations = self.analysis_results.get('correlations', {})
            for feature, stats in correlations.items():
                if abs(stats['pearson_r']) > 0.3:
                    high_corr_features.append(feature)
            
            if high_corr_features:
                print(f"• 考慮對高相關特徵進行特徵工程: {high_corr_features[:5]}...")
        
        print("\n⚠️  注意: 請在上面的 TODO 部分添加你的數據處理代碼")
        print("修改完成後，可以調用 save_processed_data() 儲存處理後的數據")
        
        return train_processed, valid_processed, test_processed
    
    def save_processed_data(self, train_df, valid_df, test_df):
        """儲存處理後的數據"""
        print("\n=== 儲存處理後的數據 ===")
        
        # 建立目錄
        processed_dir = Path("../Dataset/processed")
        processed_dir.mkdir(exist_ok=True)
        
        # 儲存
        train_df.to_csv(processed_dir / "train_processed_v2.csv", index=False, encoding='utf-8-sig')
        valid_df.to_csv(processed_dir / "valid_processed_v2.csv", index=False, encoding='utf-8-sig')
        test_df.to_csv(processed_dir / "test_processed_v2.csv", index=False, encoding='utf-8-sig')
        
        print(f"✅ 處理後數據已儲存到: {processed_dir}")
        print(f"   - train_processed_v2.csv: {train_df.shape}")
        print(f"   - valid_processed_v2.csv: {valid_df.shape}")
        print(f"   - test_processed_v2.csv: {test_df.shape}")

def main():
    """主函數"""
    print("🏠 房地產數據分析和預處理 V2")
    print("=" * 50)
    
    analyzer = DataAnalysisV2()
    
    while True:
        print("\n請選擇操作:")
        print("1. 載入數據")
        print("2. 分析數據關係")
        print("3. 自定義數據處理")
        print("0. 退出")
        
        try:
            choice = input("\n請輸入選擇 (0-3): ").strip()
            
            if choice == '0':
                print("👋 程序結束")
                break
            elif choice == '1':
                analyzer.load_data()
            elif choice == '2':
                if analyzer.train_df is None:
                    print("❌ 請先載入數據!")
                    continue
                analyzer.analyze_data_relationship()
            elif choice == '3':
                if analyzer.train_df is None:
                    print("❌ 請先載入數據!")
                    continue
                
                # 如果還沒有分析結果，先進行分析
                if not analyzer.analysis_results:
                    print("📊 先進行數據關係分析...")
                    analyzer.analyze_data_relationship(save_results=False)
                
                processed_data = analyzer.process_data_custom()
                if processed_data:
                    save_choice = input("\n是否儲存處理後的數據? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        analyzer.save_processed_data(*processed_data)
            else:
                print("❌ 無效選擇，請重新輸入")
                
        except KeyboardInterrupt:
            print("\n👋 程序已中止")
            break
        except Exception as e:
            print(f"❌ 執行錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()