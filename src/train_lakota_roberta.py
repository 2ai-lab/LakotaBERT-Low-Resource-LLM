import time
import argparse
import torch
from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def main(args):
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 1. Load Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_dir, max_len=512)

    # 2. Define Model Configuration
    config = RobertaConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12, 
        num_hidden_layers=args.num_layers,
        type_vocab_size=2,
    )

    model = RobertaForMaskedLM(config=config)
    print(f"Model parameters: {model.num_parameters()}")

    # 3. Load Datasets
    print("Loading datasets...")
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.train_file,
        block_size=128,
    )

    eval_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.val_file,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps", 
        eval_steps=1000,
        num_train_epochs=args.epochs,
        logging_steps=1000,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        save_steps=3000,
        save_total_limit=10,
        logging_dir='./logs',
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 6. Train the Model
    print("Starting training...")
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

    # 7. Save the Final Model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LakotaBERT Masked Language Model")
    
    # Path Arguments
    parser.add_argument("--tokenizer_dir", type=str, default="./model27", help="Path to the pre-trained tokenizer")
    parser.add_argument("--train_file", type=str, default="./data/train.txt", help="Path to the training data")
    parser.add_argument("--val_file", type=str, default="./data/val.txt", help="Path to the validation data")
    parser.add_argument("--output_dir", type=str, default="./output_model", help="Directory to save the trained model")
    
    # Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=52000, help="Vocabulary size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of hidden layers") #12
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)
