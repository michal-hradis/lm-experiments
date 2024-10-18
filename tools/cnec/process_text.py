from unsloth import FastLanguageModel
import torch
import argparse
from tqdm import tqdm
import logging
from prepare_cnec import prepare_prompt

def parseargs():
    parser = argparse.ArgumentParser('Parse the jsonl files and extract the text.')
    parser.add_argument('-i', '--input', required=True, help='Input jsonl file.')
    parser.add_argument('-o', '--output', required=True, help='Output text file.')
    parser.add_argument('--model', required=True, help='Model path.')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    logging.basicConfig(level=logging.INFO)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model, # or choose "unsloth/Llama-3.2-1B"
        max_seq_length = 8192,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    model.config.torch_dtype = torch.bfloat16 # Otherwise, it crashes due to the value being a string "bfloat16"

    f_out = open(args.output, 'w', encoding='utf-8')
    f_in = open(args.input, 'r', encoding='utf-8')

    for i, line in tqdm(enumerate(f_in)):
        prefix = prepare_prompt(line)
        inputs = tokenizer(
        [
            prefix
        ], return_tensors = "pt").to("cuda")
        input_length = inputs['input_ids'].shape[1]

        outputs = model.generate(**inputs, max_new_tokens=input_length * 1.3, use_cache=True)
        output = tokenizer.batch_decode(outputs)[0]
        # Get text after "entity:"
        output = output.split('entity:')[1]
        if '<|EOS|>' not in output:
            logging.warning(f'No <|EOS|> in output: {output}')
        output = output.split('<|EOS|>')[0]
        output = ' '.join(output.split())
        f_out.write(output + '\n')


if __name__ == '__main__':
    main()



