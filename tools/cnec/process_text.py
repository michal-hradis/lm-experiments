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


def process_output(output: str):
    # Get text after "entity:"
    output = output.split('entity:')[1]
    if '<|EOS|>' not in output:
        logging.warning(f'No <|EOS|> in output: {output}')
    output = output.split('<|EOS|>')[0]
    output = ' '.join(output.split())

    # add spaces before all interpunctions if there is no space before them
    interpunctions = ['.', ',', '!', '?', ':', ';', ')', ']', '}', '“', '„', '”']
    for interpunction in interpunctions:
        output = output.replace(interpunction, f' {interpunction}')

    # remove multiple spaces
    output = ' '.join(output.split())

    return output

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
    with open(args.input, 'r', encoding='utf-8') as f_in:
        input_lines = [line.strip() for line in f_in.readlines()]

    batch_size = 8
    for i in tqdm(range(0, len(input_lines), batch_size)):
        batch_lines = input_lines[i:i+batch_size]
        prefixes = [prepare_prompt(line) for line in batch_lines]
        inputs = tokenizer(prefixes, return_tensors = "pt").to("cuda")
        input_length = inputs['input_ids'].shape[1]
        outputs = model.generate(**inputs, max_new_tokens=input_length * 1.6 + 20, use_cache=True)
        outputs = tokenizer.batch_decode(outputs)
        for output in outputs:
            output = process_output(output)
            f_out.write(output + '\n')


if __name__ == '__main__':
    main()



