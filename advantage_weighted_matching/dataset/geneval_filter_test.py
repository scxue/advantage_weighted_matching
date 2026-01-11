import json

def filter_train_metadata(test_path, train_path, output_path):
    # read all prompts in test file
    test_prompts = set()
    with open(test_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            test_prompts.add(data['prompt'])
    
    # filter train file and write to new file
    with open(train_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            data = json.loads(line.strip())
            if data['prompt'] not in test_prompts:
                fout.write(line)  # keep original line, including newline

# example usage
if __name__ == "__main__":
    tasks = ['color_attr','colors','counting','single_object','two_object']
    for task in tasks:
        filter_train_metadata(
            f'/dataset/geneval/{task}/test_metadata.jsonl',
            f'/dataset/geneval/{task}/train_metadata_nofiltered.jsonl',
            f'/dataset/geneval/{task}/train_metadata.jsonl'
        )