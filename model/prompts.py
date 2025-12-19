SPECIAL_TOKENS = ["<|role|>","<|scene|>","<|context|>","<|player|>","<|assistant|>","<|sep|>"]
def build_prompt(role, scene, context, player_text):
    role = role or "未知角色"; scene = scene or "默认场景"
    context = context or ""; player_text = player_text or ""
    return (f"<|role|>{role}<|sep|>"
            f"<|scene|>{scene}<|sep|>"
            f"<|context|>{context}<|sep|>"
            f"<|player|>{player_text}\n"
            f"<|assistant|>")
