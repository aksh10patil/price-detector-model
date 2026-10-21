#!/usr/bin/env python3
"""
CLI script to train the PyTorch Deep Neural Network with Residual Blocks.
Corresponds to Week 6 redemption_train.ipynb.
"""

import argparse
import os
import torch
from dotenv import load_dotenv
from pricer.items import Item
from pricer.deep_neural_network import DeepNeuralNetworkRunner

load_dotenv(override=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep Neural Network for Price Prediction")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ed-donner/items_lite",
        help="Hugging Face dataset identifier (e.g., ed-donner/items_lite or ed-donner/items_full)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--val_size", type=int, default=1000, help="Validation set size")
    parser.add_argument("--save_path", type=str, default="deep_neural_network.pth", help="Output weights path")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset: {args.dataset}...")
    train, val, test = Item.from_hub(args.dataset)
    print(f"Loaded {len(train):,} train, {len(val):,} val, {len(test):,} test items.")

    runner = DeepNeuralNetworkRunner(train, val[: args.val_size])
    print(f"Training DNN for {args.epochs} epochs...")
    runner.train(epochs=args.epochs)

    print(f"Saving weights to {args.save_path}...")
    runner.save(args.save_path)
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
