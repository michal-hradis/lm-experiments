import argparse
from openai import OpenAI
import logging
import os
from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process texts with OpenAI completion API. The format is always: '
                    '[{"role": "system", "content": "[prompt]"}, {"role": "user", "content": "The text to analyze and process is: [text]"}]')
    parser.add_argument('--api-key', type=str, help='OpenAI API key.')
    parser.add_argument('--input', type=str, required=True, help='Input directory, file or text.')
    parser.add_argument('--output-directory', type=str, help='Output directory.')
    parser.add_argument('--output-file', type=str, help='Output file.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt. ')
    parser.add_argument('--model', type=str, required=False, default='gpt-3.5-turbo', help='Model name.')
    parser.add_argument('--json-response', action='store_true', help='Output JSON response.')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature')
    parser.add_argument('--top-p', type=float, default=0, help='Top p')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens')
    parser.add_argument('--log', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level')
    return parser.parse_args()


class GPT:
    def __init__(self, api_key, model, temperature, top_p, max_tokens, prompt):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("OpenAI API key not provided. Please provide it as an argument `--api-key` "
                             "or set the OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.prompt = prompt

    def process_text(self, text):
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"The text to analyze and process is: {text}"}
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )

        return completion.choices[0].message.content


def main():
    args = parse_args()
    logging.basicConfig(level=args.log)

    gpt = GPT(api_key=args.api_key, model=args.model, temperature=args.temperature, top_p=args.top_p,
              max_tokens=args.max_tokens, prompt=args.prompt)

    texts = []
    if os.path.isdir(args.input):
        files = glob(os.path.join(args.input, '*.txt'))
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                texts.append((file, f.readlines()))
    elif os.path.isfile(args.input):
        with open(args.input, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                texts.append((i, line))
    else:
        texts.append((1, args.input))

    texts = list(texts)
    print(texts)
    for i, text in tqdm(texts):
        response = gpt.process_text(text)
        if args.output_directory:
            output_file = os.path.join(args.output_directory, f'{i}.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
        if args.output_file:
            with open(args.output_file, 'a', encoding='utf-8') as f:
                f.write(f"{i} <<<< {response}\n")
        print(f"{i} <<<< {response}")


if __name__ == '__main__':
    main()



