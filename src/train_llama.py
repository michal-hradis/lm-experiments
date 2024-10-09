import numpy as np
import torch
import argparse
import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
import evaluate
from transformers import TrainingArguments, TextStreamer
from unsloth import is_bfloat16_supported

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="unsloth/Llama-3.2-1B", help="Either Hugging Face model name or path to model directory.")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Number of steps to accumulate gradients.")
    parser.add_argument("--max-steps", type=int, default=20000, help="Maximum number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=0.00005, help="Learning rate for training.")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory for checkpoints.")
    parser.add_argument("--seed", type=int, default=3408, help="Random seed for reproducibility.")
    parser.add_argument("--save-steps", type=int, default=500, help="Number of steps to save checkpoints.")

    parser.add_argument('--train-data', type=str, default='data.txt', help='Path to training data file.')
    parser.add_argument('--val-data', type=str, help='Path to validation data file.')
    parser.add_argument('--val-size', type=int, default=512, help='Number of samples to use for validation.')
    parser.add_argument('--num-proc', type=int, default=6, help='Number of processes to use for data loading.')


    parser.add_argument('--packing', action='store_true', help='Whether to pack sequences into a single tensor.')
    parser.add_argument('--warmup-steps', type=int, default=20, help='Number of warmup steps.')
    parser.add_argument('--lr-scheduler-type', type=str, default='linear', help='Type of learning rate scheduler.')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for optimizer.')
    return parser.parse_args()


def load_text_dataset(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return Dataset.from_dict({"text": lines})

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    if isinstance(logits, tuple):
        logits = logits[0]
    # Shift logits and labels to align correctly for cross-entropy calculation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute the loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Perplexity is the exponential of the loss
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item()}

def main():
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = 8192,
        dtype = None,
        load_in_4bit = False,
    )

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    train_dataset = load_text_dataset(args.train_data)
    val_dataset = load_text_dataset(args.val_data) if args.val_data else None

    metric = evaluate.load("perplexity")
    args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=3408,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_safetensors=False,
        eval_steps=args.save_steps,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset[:args.val_size],
        compute_metrics = metric,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 6,
        packing = True, # Can make training 5x faster for short sequences.
        args = args,
    )
    print("Finished training!")
    trainer_stats = trainer.train()

    # alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    model.config.torch_dtype = torch.bfloat16 # Otherwise, it crashes due to the value being a string "bfloat16"
    from transformers import TextStreamer

    print("Generating completions on validation set...")
    for i in np.random.randint(0, len(val_dataset), 10):
        inputs = tokenizer(
        [
            val_dataset[i]["text"]
        ], return_tensors = "pt").to("cuda")

        #outputs = model.generate(**inputs, max_new_tokens = max_seq_length, use_cache = True)
        #print(tokenizer.batch_decode(outputs))
        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048)
        print("\n")

if __name__ == "__main__":
    main()