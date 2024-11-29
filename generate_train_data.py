from datasets import load_dataset
from transformers import BertTokenizer

# Step 1: Load the dataset from Hugging Face Datasets library
def load_and_process_dataset(dataset_name, dataset_config, split="train"):
    # Load dataset (can change dataset name and config here)
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Print dataset info to help with selection
    print(f"Dataset loaded: {dataset_name}")
    print(dataset)
    
    # Get the specific split (train, validation, test)
    return dataset[split]

# Step 2: Tokenize the dataset using BERT tokenizer
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

# Step 3: Convert tokenized data to text and save to file
def save_to_file(tokenized_datasets, output_file):
    # Open the output file for writing
    with open(output_file, 'w') as f:
        for example in tokenized_datasets:
            # Convert token IDs back to text tokens
            tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
            # Write tokenized sentence into the file
            f.write(" ".join(tokens) + "\n")
    
    print(f"Training data saved to {output_file}")

if __name__ == "__main__":
    # Configuration
    dataset_name = "wikitext"  # Change dataset name here
    dataset_config = "wikitext-2-raw-v1"  # Change config if necessary
    split = "train"  # Can be 'train', 'validation', or 'test'
    output_file = "training_data.txt"  # Output file path

    # Load and process dataset
    print(f"Loading {dataset_name} dataset...")
    dataset = load_and_process_dataset(dataset_name, dataset_config, split)
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the dataset
    print("Tokenizing the dataset...")
    tokenized_datasets = tokenize_dataset(dataset, tokenizer)
    
    # Save the tokenized data to the output file
    save_to_file(tokenized_datasets, output_file)
