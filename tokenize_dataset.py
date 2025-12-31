#!/usr/bin/env python3
"""
Tokenize raw text dataset and save as HuggingFace dataset with input_ids.
Uses fast (Rust) tokenizer with multiprocessing for speed.

Usage:
    python tokenize_dataset.py --target_tokens 1_000_000_000  # for 1B tokens
"""
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize dataset for distillation training")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="robbiegwaldd/dclm-10B",
        help="HuggingFace dataset name"
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
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for parallel tokenization (default: all CPUs)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for tokenization (larger = faster but more memory)"
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: auto-estimate from target_tokens)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Get number of CPUs if not specified
    if args.num_proc is None:
        args.num_proc = os.cpu_count()

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # Verify we're using the fast (Rust) tokenizer
    if not tokenizer.is_fast:
        print("⚠️  Warning: Fast tokenizer not available, falling back to slow tokenizer")
    else:
        print(f"✅ Using fast (Rust) tokenizer")

    # Estimate how many docs we need (~400 tokens/doc average)
    if args.max_docs is None:
        args.max_docs = int(args.target_tokens / 400)
    print(f"Will process up to {args.max_docs:,} documents")

    print(f"Loading dataset: {args.dataset_name} (non-streaming)")

    # Load directly with select for efficiency
    dataset = load_dataset(
        args.dataset_name,
        split=f"{args.dataset_split}[:{args.max_docs}]",  # Use split slicing
    )
    print(f"Loaded {len(dataset):,} documents")

    # Define tokenization function
    text_field = args.text_field
    def tokenize_function(examples):
        return tokenizer(
            examples[text_field],
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )

    # Tokenize with multiprocessing
    print(f"\nTokenizing with {args.num_proc} processes (batch_size={args.batch_size})...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,  # Remove text, keep only input_ids
        desc="Tokenizing",
    )

    # Count total tokens (fast path using Arrow)
    print("\nCounting tokens...")
    import pyarrow as pa
    chunked_array = tokenized_dataset.data.column("input_ids")
    total_tokens = sum(len(chunk.flatten()) for chunk in chunked_array.chunks)
    print(f"Total tokens: {total_tokens:,}")

    # If we don't have enough tokens, warn user
    if total_tokens < args.target_tokens:
        print(f"⚠️  Warning: Only got {total_tokens:,} tokens, target was {args.target_tokens:,}")
        print(f"   Consider increasing --max_docs")

    # Save tokenized dataset
    print(f"\nSaving to {args.output_path}...")
    tokenized_dataset.save_to_disk(args.output_path)

    print(f"\n{'='*60}")
    print(f"✅ SUCCESS! Dataset saved to: {args.output_path}")
    print(f"{'='*60}")
    print(f"\nStatistics:")
    print(f"  Documents: {len(tokenized_dataset):,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"\nNext steps:")
    print(f"  1. Chunk for stage 1 & 2:")
    print(f"     python preprocess_chunk.py \\")
    print(f"         --tokenized_dataset_path {args.output_path} \\")
    print(f"         --context_length 512")
    print(f"  ")
    print(f"  2. Chunk for stage 3:")
    print(f"     python preprocess_chunk.py \\")
    print(f"         --tokenized_dataset_path {args.output_path} \\")
    print(f"         --context_length 4096")


if __name__ == "__main__":
    main()
