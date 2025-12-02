# 昇思模型开发挑战赛（S1赛季)--MultiModal赛题

根据官网的赛事指导视频研究，去替换模型里的算子

尝试了不同的模型算子替换，最终更换的是

llama的swiglu算子

qwen的rotary position embedding 和 swiglu算子

并对算子做了一些适配