from peft import PeftModel

def build_model_and_tokenizer(model_name: str, use_lora: bool = True, adapter_name: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if hasattr(model, "config") and getattr(model.config, "loss_type", None) is None:
        model.config.loss_type = "ForCausalLM"

    if use_lora and adapter_name:
        model = PeftModel.from_pretrained(model, adapter_name)

    return model, tok
