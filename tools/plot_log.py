import json
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot loss and gradient norm from JSONL file")
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the plot')
    return parser.parse_args()


# Function to read the JSONL file and extract relevant data
def read_jsonl_file(filename):
    loss_list = []
    grad_norm_list = []

    with open(filename, 'r') as file:
        for line in file:
            if not 'grad_norm' in line:
                continue
            # The file uses single quotes, which is not valid JSON
            # So we need to replace single quotes with double quotes
            line = line.replace("'", '"')
            data = json.loads(line.strip())
            print(data)
            loss_list.append(data['loss'])
            grad_norm_list.append(data['grad_norm'])

    return loss_list, grad_norm_list


# Function to plot loss and grad_norm in a single image and save as PNG
def plot_loss_grad_norm(loss_list, grad_norm_list, output_filename):
    epochs = range(len(loss_list))

    plt.figure(figsize=(10, 6))

    # Plotting gradient norm
    #plt.loglog(epochs, grad_norm_list, label="Gradient Norm", color='orange')

    # Plotting loss
    plt.semilogy(epochs, loss_list, label="Loss", color='blue')

    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title('Loss and Gradient Norm Over Time')
    plt.legend()

    # Save the plot to a PNG file
    plt.savefig(output_filename, format='png')
    plt.close()


def main():
    args = parse_args()
    loss_data, grad_norm_data = read_jsonl_file(args.input)
    plot_loss_grad_norm(loss_data, grad_norm_data, args.output)
    print(f"Plot saved as {args.output}")


if __name__ == '__main__':
    main()

