from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import time
import torch
import sys
import platform

class GenerationProgressCallback(StoppingCriteria):
    def __init__(self, interval=10):
        self.interval = interval 
        self.token_count = 0
        self.start_time = None
    
    def __call__(self, input_ids, scores, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
        
        self.token_count += 1
        if self.token_count % self.interval == 0:
            elapsed = time.time() - self.start_time
            speed = self.token_count / elapsed if elapsed > 0 else 0
            print(f"[GENERATE] Token {self.token_count} | {speed:.1f} tok/s", flush=True)
        return False 

def generate(model_name, content, system_prompt, max_new_tokens, cache_dir=None, use_bitsandbytes = True):
    print(f"[GENERATE] Loading tokenizer: {model_name}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"[GENERATE] Tokenizer loaded ({time.time() - start_time:.2f}s)")
    
    print(f"[GENERATE] Loading model from cache_dir: {cache_dir or 'default'}")
    start_time = time.time()
    
    has_gpu = torch.cuda.is_available()
    is_windows = platform.system() == 'Windows'
    
    if has_gpu:
        print(f"[GENERATE] GPU detected: {torch.cuda.get_device_name(0)}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
    elif is_windows and not has_gpu and use_bitsandbytes:
    
        print(f"[GENERATE] Windows CPU mode - using 4-bit quantization")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=cache_dir,
                low_cpu_mem_usage=True
            )
        except ImportError:
            print(f"[GENERATE] bitsandbytes not available, loading without quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                cache_dir=cache_dir,
                low_cpu_mem_usage=True
            )
    elif is_windows and not has_gpu and not use_bitsandbytes:   
        print(f"[GENERATE] bitsandbytes disabled, loading without quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            cache_dir=cache_dir,
            low_cpu_mem_usage=True
        )     
    else:
        print(f"[GENERATE] CPU mode on Linux")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            cache_dir=cache_dir,
            low_cpu_mem_usage=True
        )
    
    print(f"[GENERATE] Model loaded ({time.time() - start_time:.2f}s)")
    print(f"[GENERATE] Device: {next(model.parameters()).device}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_tokens = model_inputs.input_ids.shape[1]
    print(f"[GENERATE] Input tokens: {input_tokens}")
    
    print(f"[GENERATE] Generating (max_new_tokens={max_new_tokens})...")
    callback = GenerationProgressCallback(interval=10)
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        stopping_criteria=StoppingCriteriaList([callback])
    )
    generation_time = time.time() - start_time
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_tokens = len(generated_ids[0])
    
    print(f"[GENERATE] Generated {output_tokens} tokens in {generation_time:.2f}s ({output_tokens/generation_time:.1f} tok/s)")
    
    return response

