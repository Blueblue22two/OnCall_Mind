"""快速验证 eval_judge_api_base 是否可正常调用"""
import sys
sys.path.insert(0, ".")

from openai import OpenAI
from app.config import config

client = OpenAI(
    api_key=config.eval_judge_api_key,
    base_url=config.eval_judge_api_base,
)

print(f"API Base: {config.eval_judge_api_base}")
print(f"Model:    {config.eval_judge_model}")

# 1. 先测 /models 确认鉴权
try:
    models = client.models.list()
    model_ids = [m.id for m in models]
    print(f"Models available: {len(model_ids)} (first 5: {model_ids[:5]})")
except Exception as e:
    print(f"[FAIL] /models: {e}")
    sys.exit(1)

# 2. 测 chat completion 确认模型可用
try:
    resp = client.chat.completions.create(
        model=config.eval_judge_model,
        messages=[{"role": "user", "content": "回复 OK"}],
        temperature=config.eval_judge_temperature,
        max_tokens=16,
    )
    content = resp.choices[0].message.content
    print(f"Chat response: {content}")
    print("[PASS] eval_judge_api_base 调用正常")
except Exception as e:
    print(f"[FAIL] chat completion: {e}")
    sys.exit(1)
