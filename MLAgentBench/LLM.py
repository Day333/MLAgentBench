""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
import tiktoken
from .schema import TooLongPromptError, LLMError

enc = tiktoken.get_encoding("cl100k_base")

try:
    import anthropic
    # setup anthropic API key
    anthropic_client = anthropic.Anthropic(api_key=open("claude_api_key.txt").read().strip())
except Exception as e:
    print(e)
    print("Could not load anthropic API key claude_api_key.txt.")

# anthropic-sdk-python >=1.0 removed the legacy HUMAN_PROMPT/AI_PROMPT constants along with
# the old Completions API. They are used here purely as conversation-turn text markers, so we
# restore their historical values for compatibility with modern SDK versions.
HUMAN_PROMPT = getattr(anthropic, "HUMAN_PROMPT", "\n\nHuman:")
AI_PROMPT = getattr(anthropic, "AI_PROMPT", "\n\nAssistant:")

try:
    import openai
    # setup OpenAI API key
    openai.organization, openai.api_key  =  open("openai_api_key.txt").read().strip().split(":")
    os.environ["OPENAI_API_KEY"] = openai.api_key
except Exception as e:
    print(e)
    print("Could not load OpenAI API key openai_api_key.txt.")

try:
    import openai
    # setup Qwen (Alibaba Cloud Model Studio / DashScope OpenAI-compatible) client
    qwen_client = openai.OpenAI(
        api_key=open("qwen_api_key.txt").read().strip(),
        base_url="https://ws-r1de3xs467h5tod7.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
except Exception as e:
    print(e)
    print("Could not load Qwen API key qwen_api_key.txt.")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

loaded_hf_models = {}

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids, scores, **kwargs):
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype)

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())

    
def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")

def complete_text_hf(prompt, stop_sequences=[], model="huggingface/codellama/CodeLlama-7b-hf", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
    model = model.split("/", 1)[1]
    if model in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model]
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model).to("cuda:9")
        tokenizer = AutoTokenizer.from_pretrained(model)
        loaded_hf_models[model] = (hf_model, tokenizer)
        
    encoded_input = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:9")
    stop_sequence_ids = tokenizer(stop_sequences, return_token_type_ids=False, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList()
    for stop_sequence_input_ids in stop_sequence_ids.input_ids:
        stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))

    output = hf_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = stopping_criteria,
        **kwargs,
    )
    sequences = output.sequences
    sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences)
    completion = all_decoded_text[0]
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text_claude(prompt, stop_sequences=[HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, messages=None, **kwargs):
    """ Call the Claude API to complete a prompt."""

    ai_prompt = AI_PROMPT
    if "ai_prompt" in kwargs is not None:
        ai_prompt = kwargs["ai_prompt"]

    
    try:
        if model == "claude-3-opus-20240229":
            while True:
                try:
                    message = anthropic_client.messages.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ] if messages is None else messages,
                        model=model,
                        stop_sequences=stop_sequences,
                        temperature=temperature,
                        max_tokens=max_tokens_to_sample,
                        **kwargs
                    )
                except anthropic.InternalServerError as e:
                    pass
                try:
                    completion = message.content[0].text
                    break
                except:
                    print("end_turn???")
                    pass
        else:
            rsp = anthropic_client.completions.create(
                prompt=f"{HUMAN_PROMPT} {prompt} {ai_prompt}",
                stop_sequences=stop_sequences,
                model=model,
                temperature=temperature,
                max_tokens_to_sample=max_tokens_to_sample,
                **kwargs
            )
            completion = rsp.completion
        
    except anthropic.APIStatusError as e:
        print(e)
        raise TooLongPromptError()
    except Exception as e:
        raise LLMError(e)

    
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text_openai(prompt, stop_sequences=[], model="gpt-3.5-turbo", max_tokens_to_sample=500, temperature=0.2, log_file=None, **kwargs):
    """ Call the OpenAI API to complete a prompt."""
    raw_request = {
          "model": model,
          "temperature": temperature,
          "max_tokens": max_tokens_to_sample,
          "stop": stop_sequences or None,  # API doesn't like empty list
          **kwargs
    }
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(**{"messages": messages,**raw_request})
        completion = response["choices"][0]["message"]["content"]
    else:
        response = openai.Completion.create(**{"prompt": prompt,**raw_request})
        completion = response["choices"][0]["text"]
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text_qwen(prompt, stop_sequences=[], model="qwen-plus", max_tokens_to_sample=2000, temperature=0.5, log_file=None, **kwargs):
    """ Call the Qwen API (Alibaba Cloud Model Studio, OpenAI-compatible mode) to complete a prompt."""
    messages = [{"role": "user", "content": prompt}]
    response = qwen_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens_to_sample,
        stop=stop_sequences or None,
        **kwargs
    )
    completion = response.choices[0].message.content
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    
    if model.startswith("claude"):
        # use anthropic API
        completion = complete_text_claude(prompt, stop_sequences=[HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    elif model.startswith("huggingface"):
        completion = complete_text_hf(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    elif model.startswith("qwen"):
        completion = complete_text_qwen(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        # use OpenAI API
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    return completion

# specify fast models for summarization etc
FAST_MODEL = "claude-v1"
def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)
# complete_text_fast = partial(complete_text_openai, temperature= 0.01)

