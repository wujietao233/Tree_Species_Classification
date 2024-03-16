"""
评估模型
"""
from utils import test_model

result = test_model("weights/mobilenetv2_elu/BarkVN-50/2024-01-14_11.55.14")
print(result)