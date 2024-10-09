import numpy as np
from unsloth import FastLanguageModel
import torch
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/mnt/data3/checkpoints_2/checkpoint-11500", # or choose "unsloth/Llama-3.2-1B"
    max_seq_length = 8192,
    dtype = torch.bfloat16,
    load_in_4bit = False,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
model.config.torch_dtype = torch.bfloat16 # Otherwise, it crashes due to the value being a string "bfloat16"
from transformers import TextStreamer

print("Generating completions...")
while True:
    prefix = input("Write input: ")
    inputs = tokenizer(
    [
        prefix
    ], return_tensors = "pt").to("cuda")

    #outputs = model.generate(**inputs, max_new_tokens = max_seq_length, use_cache = True)
    #print(tokenizer.batch_decode(outputs))
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512)
    print("\n")


print("Generating completions while streaming...")

# ------------------------------------------------------------------------------
#text_streamer = TextStreamer(tokenizer)
#_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048)

