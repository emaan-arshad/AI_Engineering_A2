"""Verify model IDs against OpenRouter before benchmarking (assignment: 'verify the current model pages').

    python -m scripts.check_models

The public /api/v1/models listing may not include Decisions-only models; a miss here is a hint,
not proof. The final check is the Warm-up suite run.
"""
import httpx

from backend.adapters import MODEL_REGISTRY

ids = {m["id"] for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json().get("data", [])}
for key, (label, _cat, model_id, iface) in MODEL_REGISTRY.items():
    print(f"{'FOUND  ' if model_id in ids else 'MISSING'}  {model_id:36s} ({label}, {iface})")
    if model_id not in ids:
        near = sorted(i for i in ids if model_id.split('/')[-1].split('-')[0] in i)[:5]
        if near:
            print("         similar ids:", ", ".join(near))
