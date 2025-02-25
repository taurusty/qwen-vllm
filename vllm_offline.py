from vllm_wrapper import vLLMWrapper
import torch

# 设置CUDA内存分配参数
torch.cuda.set_per_process_memory_fraction(0.7)  # 增加GPU内存使用比例
torch.backends.cudnn.benchmark = True

model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

vllm_model = vLLMWrapper(model,
                         # 使用实际支持的数据类型
                         dtype="float16",  # 改回标准数据类型
                         tensor_parallel_size=1,
                         gpu_memory_utilization=0.7,  # 增加内存使用率
                         max_model_len=1024,
                         model_type="deepseek")  # 显式指定模型类型

history=None 
system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."

while True:
    Q=input('提问:')
    response, history = vllm_model.chat(query=Q,
                                      history=history,
                                      system=system_prompt)
    print(response)
    history=history[:20]