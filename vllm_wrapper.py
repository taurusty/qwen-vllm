import copy
# 设置环境变量必须在导入vLLM之前
import os
os.environ['VLLM_USE_MODELSCOPE'] = 'True'

from vllm import LLM
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig, snapshot_download
# 只导入remove_stop_words，不导入_build_prompt
from prompt_utils import remove_stop_words

# 通义千问的特殊token
IMSTART='<|im_start|>'  
IMEND='<|im_end|>'
ENDOFTEXT='<|endoftext|>'     # EOS以及PAD都是它

class vLLMWrapper:
    def __init__(self, 
                 model_dir,
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.90,
                 dtype='float16',
                 quantization=None,
                 max_model_len=None,
                 model_type=None):
        # 模型目录下的generation_config.json文件，是推理的关键参数
        '''
        {
            "chat_format": "chatml",
            "eos_token_id": 151643,
            "pad_token_id": 151643,
            "max_window_size": 6144,
            "max_new_tokens": 512,
            "do_sample": true,
            "top_k": 0,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "transformers_version": "4.31.0"
            }
        '''
        
        # 模型下载并获取本地路径
        model_path = snapshot_download(model_dir)
        print(f"模型已下载到: {model_path}")
        
        # 从本地路径加载配置
        self.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id
        
        # 确定模型类型并设置相应的停止词
        if model_type is not None:
            self.model_type = model_type
        else:
            self.model_type = self._detect_model_type(model_dir)
        
        print(f"使用模型类型: {self.model_type}")
        self.stop_words_ids = self._get_stop_words_ids()
        
        # vLLM加载模型（使用本地路径）
        model_args = {
            "model": model_path,
            "tokenizer": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "quantization": quantization,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype
        }
        
        # 如果提供了max_model_len，添加到参数中
        if max_model_len is not None:
            model_args["max_model_len"] = max_model_len
            
        self.model = LLM(**model_args)

    def _detect_model_type(self, model_dir):
        """检测模型类型"""
        model_dir_lower = model_dir.lower()
        
        # 特殊情况：DeepSeek-Qwen 混合模型
        if "deepseek" in model_dir_lower and "qwen" in model_dir_lower:
            print("检测到 DeepSeek-Qwen 混合模型，使用 DeepSeek 格式")
            return "deepseek"
        
        # 一般情况
        if "deepseek" in model_dir_lower:
            return "deepseek"
        elif "qwen" in model_dir_lower:
            return "qwen"
        else:
            # 默认使用通用设置
            return "generic"
    
    def _get_stop_words_ids(self):
        """根据模型类型获取停止词ID"""
        if self.model_type == "qwen":
            # Qwen模型特殊标记
            if hasattr(self.tokenizer, 'im_start_id') and hasattr(self.tokenizer, 'im_end_id'):
                return [self.tokenizer.im_start_id, self.tokenizer.im_end_id, self.tokenizer.eos_token_id]
        
        # 对于其他模型，只使用eos_token_id
        return [self.tokenizer.eos_token_id]

    def chat(self, query, history=None, system="You are a helpful assistant.", extra_stop_words_ids=[]):
        # 添加调试输出
        print(f"当前模型类型: {self.model_type}")
        
        # 历史聊天
        if history is None:
            history = []
        else:
            history = copy.deepcopy(history)

        # 额外指定推理停止词
        stop_words_ids = self.stop_words_ids + extra_stop_words_ids

        # 根据模型类型构造不同的prompt
        if self.model_type == "deepseek":
            # DeepSeek模型使用不同的提示格式
            prompt_text = self._build_deepseek_prompt(query, history, system)
            print(f"使用DeepSeek提示格式: {prompt_text[:100]}...")
            prompt_tokens = self.tokenizer.encode(prompt_text)
        else:
            # 只有Qwen模型才使用原有的_build_prompt函数
            from prompt_utils import _build_prompt
            prompt_text, prompt_tokens = _build_prompt(self.generation_config, self.tokenizer, 
                                                    query, history=history, system=system)
        
        # 打开注释，观测底层Prompt构造
        # print(prompt_text)

        # VLLM请求配置
        sampling_params=SamplingParams(stop_token_ids=stop_words_ids, 
                                         top_p=self.generation_config.top_p,
                                         top_k=-1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
                                         temperature=self.generation_config.temperature,
                                         repetition_penalty=self.generation_config.repetition_penalty,
                                         max_tokens=self.generation_config.max_new_tokens)
        
        # 调用VLLM执行推理，使用文本输入而不是token IDs
        req_outputs=self.model.generate(prompts=[prompt_text], sampling_params=sampling_params, use_tqdm=False)
        req_output=req_outputs[0]    
        
        # transformer模型的原生返回, 打开注释看一下原始推理结果
        # raw_response=req_output.outputs[0].text
        # print(raw_response)
        
        # 移除停用词        
        response_token_ids=remove_stop_words(list(req_output.outputs[0].token_ids), stop_words_ids)
        response=self.tokenizer.decode(response_token_ids)

        # 整理历史对话
        history.append((query,response))
        return response,history

    def _build_deepseek_prompt(self, query, history, system):
        """为DeepSeek模型构建提示"""
        prompt = f"<|system|>\n{system}\n<|user|>\n{query}\n<|assistant|>\n"
        
        # 如果有历史对话，添加到提示中
        if history:
            formatted_history = ""
            for user_msg, assistant_msg in history:
                formatted_history += f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}\n"
            prompt = f"<|system|>\n{system}\n{formatted_history}<|user|>\n{query}\n<|assistant|>\n"
            
        return prompt