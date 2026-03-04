import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def main(args):
    print(f"Loading tokenizer and model from: {args.model_path}")
    
    # 1. Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)

    # 2. Load testing sentences
    print(f"Loading testing data from: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as file:
        eval_sentences = [line.strip() for line in file if line.strip()][:args.max_sentences]

    # 3. Tokenize the sentences
    encoded_sentences = tokenizer(
        eval_sentences, 
        return_tensors="pt", 
        padding=True, 
        truncation=True if args.max_length else False, 
        max_length=args.max_length
    )
    input_ids = encoded_sentences["input_ids"]
    attention_mask = encoded_sentences["attention_mask"]

    # 4. Replace a random 15% of the tokens with the mask token
    masked_input_ids = input_ids.clone()
    # Ensure we do not mask padding tokens
    indices_to_mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool() & (input_ids != tokenizer.pad_token_id)
    masked_input_ids[indices_to_mask] = tokenizer.mask_token_id

    # 5. Get outputs from the model (using no_grad to save memory)
    with torch.no_grad():
        outputs = model(masked_input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 6. Calculate accuracy
    predicted_ids = torch.argmax(logits, dim=-1)
    correct_predictions = torch.eq(predicted_ids[indices_to_mask], input_ids[indices_to_mask])
    masked_positions = indices_to_mask.sum()

    if masked_positions > 0:
        accuracy = (correct_predictions.sum().float() / masked_positions) * 100
    else:
        accuracy = torch.tensor(0.0)

    # 7. Output Results
    print("\n--- Evaluation Results ---")
    print(f"Number of correct predictions: {correct_predictions.sum().item()}")
    print(f"Total masked positions: {masked_positions.item()}")
    print(f"Accuracy (Raw Ratio): {(accuracy.item() / 100):.4f}")
    print(f"Masked Language Modeling Accuracy: {accuracy.item():.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LakotaBERT Basic MLM Accuracy")
    
    # Path Arguments
    parser.add_argument("--model_path", type=str, default="./output_model", help="Path to the trained model directory")
    parser.add_argument("--test_file", type=str, default="./data/test.txt", help="Path to the test data file")
    
    # Evaluation Parameters
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_sentences", type=int, default=100, help="Number of sentences to evaluate for efficiency")
    
    args = parser.parse_args()
    main(args)
