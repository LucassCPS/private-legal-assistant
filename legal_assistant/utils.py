from langchain_ollama import OllamaLLM

def initialize_model(model_name: str, model_temperature: float = 0.7, model_ctx: int = 4096, model_num_gpu: int = 1):
    return OllamaLLM(model=model_name, temperature=model_temperature, num_ctx=model_ctx, num_gpu=model_num_gpu)