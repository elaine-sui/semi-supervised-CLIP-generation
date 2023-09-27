from transformers import GPT2LMHeadModel, AutoModelForCausalLM

def build_huggingface_model(hf_model_name, tokenizer=None):
    if hf_model_name == "gpt2":
        return GPT2LMHeadModel.from_pretrained(hf_model_name)
    elif hf_model_name == "meta-llama/Llama-2-7b-hf":
        # import pdb; pdb.set_trace()
        return AutoModelForCausalLM.from_pretrained(hf_model_name, use_auth_token=True)
        # model.config.pad_token_id = model.config.eos_token_id
        # model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, model.config.pad_token_id)
        # model.resize_token_embeddings(len(tokenizer))
    else:
        raise NotImplementedError(
            f"Huggingface model builder not implemented for {hf_model_name}"
        )