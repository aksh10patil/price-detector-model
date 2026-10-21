"""Training script for PyTorch Deep Residual Neural Network."""

import argparse
from pricer.data.schema import Item
from pricer.models.dnn import DeepNeuralNetworkRunner
from pricer.config import DEFAULT_LITE_DATASET


def train_dnn(dataset_name: str = DEFAULT_LITE_DATASET, epochs: int = 5, save_path: str = "deep_neural_network.pth"):
    print(f"Loading dataset {dataset_name} for DNN training...")
    train, val, test = Item.from_hub(dataset_name)
    print(f"Loaded {len(train):,} train, {len(val):,} val items.")

    runner = DeepNeuralNetworkRunner(train, val[:1000])
    print(f"Starting {epochs} epochs of ResNet training...")
    runner.train(epochs=epochs)

    runner.save(save_path)
    print(f"Weights saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_LITE_DATASET)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="deep_neural_network.pth")
    args = parser.parse_args()
    train_dnn(args.dataset, args.epochs, args.save_path)
