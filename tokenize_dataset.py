#!/usr/bin/env python3
"""
Tokenize raw text dataset and save as HuggingFace dataset with input_ids.
This is the missing Step 0 before running preprocess_chunk.py.

Usage:
    python tokenize_dataset.py --target_tokens 1_000_000_000  # for 1B tokens
"""
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize dataset for distillation training")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="robbiegwaldd/dclm-10B",
        help="HuggingFace dataset name (LGizkde/dclm-downsampled)"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace tokenizer name (should match your teacher model)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/workspace/checkpoints/data_cache/tokenized_tokens.arrow",
        help="Output path for tokenized dataset"
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=10_000_000_000,
        help="Target number of tokens to collect (default: 10B)"
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Name of the text field in the dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of documents to process at once"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"Loading dataset: {args.dataset_name} (streaming mode)")
    dataset = load_dataset(
        args.dataset_name,
        # name="ppl",
        split=args.dataset_split,
        streaming=True,
        trust_remote_code=True
    )

    all_input_ids = []
    total_tokens = 0
    processed_docs = 0

    print(f"Tokenizing documents (target: {args.target_tokens:,} tokens)...")

    batch = []
    for example in tqdm(dataset, desc="Processing documents"):
        batch.append(example[args.text_field])

        # Process in batches for efficiency
        if len(batch) >= args.batch_size:
            # Tokenize batch
            tokenized = tokenizer(
                batch,
                truncation=False,
                padding=False,
                add_special_tokens=False,
                return_attention_mask=False,
            )

            # Collect all input_ids
            for input_ids in tokenized["input_ids"]:
                all_input_ids.append(input_ids)
                total_tokens += len(input_ids)

            processed_docs += len(batch)
            batch = []

            # Check if we've reached target
            if total_tokens >= args.target_tokens:
                print(f"\n✅ Reached target of {args.target_tokens:,} tokens!")
                break

            # Progress update
            if processed_docs % 10000 == 0:
                print(f"Processed {processed_docs:,} docs, {total_tokens:,} tokens")

    # Process remaining batch
    if batch and total_tokens < args.target_tokens:
        tokenized = tokenizer(
            batch,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        for input_ids in tokenized["input_ids"]:
            all_input_ids.append(input_ids)
            total_tokens += len(input_ids)
        processed_docs += len(batch)

    print(f"\nTokenization complete!")
    print(f"  Documents processed: {processed_docs:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total sequences: {len(all_input_ids):,}")

    # Create HuggingFace dataset
    print(f"\nCreating HuggingFace dataset...")
    tokenized_dataset = Dataset.from_dict({"input_ids": all_input_ids})

    print(f"Saving to {args.output_path}...")
    tokenized_dataset.save_to_disk(args.output_path)

    print(f"\n{'='*60}")
    print(f"✅ SUCCESS! Dataset saved to: {args.output_path}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Chunk for stage 1 & 2:")
    print(f"     python preprocess_chunk.py --context_length 512")
    print(f"  ")
    print(f"  2. Chunk for stage 3:")
    print(f"     python preprocess_chunk.py --context_length 4096")


if __name__ == "__main__":
    main()
