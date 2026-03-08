from datasets import load_dataset

def load_conll_dataset():
    dataset = load_dataset("conll2003")
    return dataset