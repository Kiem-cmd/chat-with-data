import torch
from huggingface_hub import login
from transformers import BitsAndBytesConfig 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
from langchain_community.llms import HuggingFacePipeline

# login(token = "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")


nf4_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16
)

def get_model(model_name = "meta-llama/Llama-3.2-3B-Instruct", max_new_token = 1024, **kwargs):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = nf4_config,
        low_cpu_mem_usage = True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model_pipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = max_new_token,
        pad_token_id = tokenizer.eos_token_id,
        device_map = "auto"
    )

    llm = HuggingFacePipeline(
        pipeline = model_pipeline,
        model_kwargs = kwargs
    )
    return llm


if __name__ == "__main__":
    question = "Who is Donald Trump ?" 
    llm = get_model()
    print(f"Question: {question}")
    print(f"Answer: {llm(question)}")