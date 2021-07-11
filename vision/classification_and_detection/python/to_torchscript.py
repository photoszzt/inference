import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = torch.load(args.input)
    model.eval()
    sm = torch.jit.script(model)
    torch.jit.save(sm, args.output)


if __name__ == '__main__':
    main()
