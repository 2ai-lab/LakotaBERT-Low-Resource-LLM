import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import precision_score, f1_score
import Levenshtein as lev
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main(args):
    print(f"Loading model and tokenizer from: {args.model_name_or_path}")
    
    # 1. Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    # 2. Load evaluation sentences
    print(f"Loading evaluation data from: {args.eval_file}")
    with open(args.eval_file, 'r', encoding='utf-8') as file:
        eval_sentences = [line.strip() for line in file if line.strip()][:args.max_sentences]

    # 3. Tokenize sentences
    encoded_sentences = tokenizer(
        eval_sentences, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=args.max_length
    )
    input_ids = encoded_sentences["input_ids"]
    attention_mask = encoded_sentences["attention_mask"]

    # 4. Mask 15% of the tokens
    masked_input_ids = input_ids.clone()
    indices_to_mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool() & (input_ids != tokenizer.pad_token_id)
    masked_input_ids[indices_to_mask] = tokenizer.mask_token_id

    # 5. Get model predictions
    with torch.no_grad():
        outputs = model(masked_input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_ids = torch.argmax(logits, dim=-1)
    masked_positions = indices_to_mask.sum().item()

    # 6. Calculate Metrics
    if masked_positions > 0:
        correct_predictions = torch.eq(predicted_ids[indices_to_mask], input_ids[indices_to_mask]).sum().item()
        accuracy = (correct_predictions / masked_positions) * 100
        
        y_true = input_ids[indices_to_mask].cpu().numpy()
        y_pred = predicted_ids[indices_to_mask].cpu().numpy()

        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Calculate MRR
        reciprocal_ranks = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
        mrr = np.mean(reciprocal_ranks)
        
        # Calculate CER
        true_tokens = tokenizer.batch_decode(y_true, skip_special_tokens=True)
        pred_tokens = tokenizer.batch_decode(y_pred, skip_special_tokens=True)

        cer_scores = []
        for true, pred in zip(true_tokens, pred_tokens):
            if len(true) > 0 or len(pred) > 0:
                cer_score = lev.distance(true, pred) / max(len(true), len(pred))
            else:
                cer_score = 0.0
            cer_scores.append(cer_score)
        cer = np.mean(cer_scores) if cer_scores else float('inf')

        # Calculate Hit@K
        k = 10
        # Correctly isolate the logits for only the masked positions before getting top K
        top_k_predictions = torch.topk(logits[indices_to_mask], k, dim=-1).indices.cpu().numpy()
        hits_at_k = np.mean([1 if true in pred else 0 for true, pred in zip(y_true, top_k_predictions)])

        # Calculate BLEU score
        smoothing_function = SmoothingFunction().method1
        bleu_scores = []
        for true, pred in zip(true_tokens, pred_tokens):
            reference = [true.split()]
            candidate = pred.split()
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)
        bleu = np.mean(bleu_scores) if bleu_scores else 0.0

        # Calculate Perplexity
        softmax_probs = torch.softmax(logits.view(-1, logits.size(-1)), dim=-1)
        true_probs = softmax_probs[torch.arange(softmax_probs.size(0)), input_ids.view(-1)].reshape(input_ids.size())
        
        if (true_probs == 0).any():
            print("Warning: Some true probabilities are zero.")

        neg_log_likelihood = -torch.log(true_probs + 1e-10)
        perplexity = torch.exp(neg_log_likelihood.sum() / masked_positions).item()

    else:
        correct_predictions, accuracy, precision, f1, mrr, hits_at_k, bleu = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        cer, perplexity = float('inf'), float('inf')

    # 7. Output Results
    print("\n--- Evaluation Results ---")
    print(f"Total masked positions: {masked_positions}")
    print(f"Number of correct predictions: {correct_predictions}")
    print(f"Masked Language Modeling Accuracy: {accuracy:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Hit@{k}: {hits_at_k:.4f}")
    print(f"BLEU Score: {bleu:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LakotaBERT Model Metrics")
    
    # Path Arguments
    parser.add_argument("--model_name_or_path", type=str, default="./output_model", help="Path to the trained LakotaBERT model or Hugging Face repo")
    parser.add_argument("--eval_file", type=str, default="./data/val.txt", help="Path to the validation/test data")
    
    # Evaluation Parameters
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for truncation")
    parser.add_argument("--max_sentences", type=int, default=100, help="Number of sentences to evaluate (for efficiency)")
    
    args = parser.parse_args()
    main(args)
