import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--x", default=1, type=int)

args = parser.parse_args()


output = {"a": 2, "x": "2"}
for k, v in output.items():
    args.__setattr__(k, v)
print(args)
print(args.a)
