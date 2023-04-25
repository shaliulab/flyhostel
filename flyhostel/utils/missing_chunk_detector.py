import sys
import re
import argparse

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", nargs="+", type=int, required=True)
    return ap

def main():

    ap = get_parser()
    args=ap.parse_args()


    # Read the input from standard input
    standard_input = sys.stdin.read()
    lines = [line.strip() for line in standard_input.split("\n")]
    lines = [line for line in lines if len(line)>0]
    

    chunks = []

    for line in lines:
        try:
            chunks.append(int(re.search(".*/session_([0-9][0-9][0-9][0-9][0-9][0-9])/.*", line).group(1)))
        except:
            pass

    if len(chunks) == 0:
        raise ValueError("No chunks found in input")


    target = list(range(*args.interval))

    missing=[]
    for chunk in target:
        if chunk not in chunks:
            missing.append(chunk)

    for chunk in missing:
        print(chunk)


    


if __name__ == "__main__":
    main()