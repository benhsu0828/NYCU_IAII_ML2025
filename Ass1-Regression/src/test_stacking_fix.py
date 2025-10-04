from model import SerializableDNNWrapper
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

# 測試 SerializableDNNWrapper 是否正確繼承
wrapper = SerializableDNNWrapper(input_dim=10)
print('✅ SerializableDNNWrapper 創建成功')
print(f'✅ 是 BaseEstimator: {isinstance(wrapper, BaseEstimator)}')
print(f'✅ 是 RegressorMixin: {isinstance(wrapper, RegressorMixin)}')
print(f'✅ 有 fit 方法: {hasattr(wrapper, "fit")}')
print(f'✅ 有 predict 方法: {hasattr(wrapper, "predict")}')
print(f'✅ 有 score 方法: {hasattr(wrapper, "score")}')
print(f'✅ 有 get_params 方法: {hasattr(wrapper, "get_params")}')
print(f'✅ 有 set_params 方法: {hasattr(wrapper, "set_params")}')

# 測試 scikit-learn 是否認為它是回歸器
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import is_regressor
print(f'✅ scikit-learn 認為它是回歸器: {is_regressor(wrapper)}')