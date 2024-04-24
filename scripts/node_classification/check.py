import json
import glob

def run(args):

    results = []
    result_paths = glob.glob(args.path + "/*/*/result.json")
    for result_path in result_paths:
        try:
            with open(result_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    result = json.loads(line)
                    results.append(result)
        except:
            pass
    
    results = sorted(results, key=lambda x: x["val/accuracy"], reverse=True)
    print(results[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="results")
    args = parser.parse_args()
    run(args)
