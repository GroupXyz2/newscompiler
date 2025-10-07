# Script only for testing, final app would have api.py instead of entry.py

import parser
import generate
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

if __name__ == "__main__":
    # some random sites i found first for testing, no representation whatsoever!
    news_sites = [
        "https://www.br.de/nachrichten",
        "https://www.t-online.de/nachrichten"
    ]
    content = parser.extract_from_sites(news_sites, max_articles_per_site=3, replacements=["Loading..."])
    
    print("=== EXTRACTED CONTENT ===")
    print(content)
    print("=====================")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens = 512
    system_prompt = f"Your task is to provide a concise overview of the given news topics. Use a maximum of {max_new_tokens} tokens in your response."
    cache_dir = r"E:\huggingface\.cache"
    response = generate.generate(model_name, content, system_prompt, max_new_tokens, cache_dir=cache_dir, use_bitsandbytes=False)

    print("=== MODEL RESPONSE ===")
    print(response)
    print("=====================")