"""
Encodes text labels using SigLIP2 model and saves the embeddings.
"""

import torch
import argparse
from pathlib import Path
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer


class SigLIPLabelEncoder:
    """Encodes text labels using SigLIP model."""

    def __init__(
        self, model_name: str = "siglip2-base-patch16-512", device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load model and tokenizer
        model_path = f"google/{model_name}"
        self.model = AutoModel.from_pretrained(model_path).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(f"Loaded {model_name} model on {self.device}")

    def encode_labels(self, labels: List[str], add_prefix: bool = True) -> torch.Tensor:
        """
        Encode a list of labels into embeddings.

        Args:
            labels: List of label strings
            add_prefix: Whether to add "this is a" prefix to labels

        Returns:
            Normalized text embeddings tensor
        """
        # Add prefix if requested
        if add_prefix:
            prompts = [f"this is a {label}" for label in labels]
        else:
            prompts = labels

        if self.model_name == "siglip-base-patch16-512":
            inputs = self.tokenizer(prompts, padding="max_length", return_tensors="pt")
        else:
            # For siglip2 and other variants
            inputs = self.tokenizer(
                prompts, padding="max_length", max_length=64, return_tensors="pt"
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )

        return text_embeddings.cpu()


def load_labels_from_file(file_path: Path) -> List[str]:
    """Load labels from a text file (one label per line)."""
    with open(file_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def parse_labels_from_string(labels_str: str) -> List[str]:
    return [label.strip() for label in labels_str.split(",") if label.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Encode text labels using SigLIP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--labels", type=str, help="Comma-separated list of labels to encode"
    )
    input_group.add_argument(
        "--labels_file",
        type=Path,
        help="Path to text file containing labels (one per line)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for saved embeddings (.pt file)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="siglip2-base-patch16-512",
        choices=["siglip-base-patch16-512", "siglip2-base-patch16-512"],
        help="SigLIP model variant to use",
    )

    parser.add_argument(
        "--no_prefix",
        action="store_true",
        help="Don't add 'this is a' prefix to labels",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: auto-detect)",
    )

    args = parser.parse_args()

    if args.labels:
        labels = parse_labels_from_string(args.labels)
        print(f"Loaded {len(labels)} labels from command line")
    else:
        labels = load_labels_from_file(args.labels_file)
        print(f"Loaded {len(labels)} labels from {args.labels_file}")

    if not labels:
        print("Error: No labels found!")
        return 1

    print(f"Labels: {labels[:5]}{'...' if len(labels) > 5 else ''}")

    encoder = SigLIPLabelEncoder(model_name=args.model, device=args.device)
    print(f"\nEncoding {len(labels)} labels...")
    embeddings = encoder.encode_labels(labels, add_prefix=not args.no_prefix)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[-1]}")
    print(f"Mean norm: {embeddings.norm(dim=-1).mean().item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    print(f"\nEmbeddings saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())

"""
Examples:

# From command line labels
python encode_labels.py --labels "chair,table,door,window" --output embeddings/furniture.pt

# From text file
python encode_labels.py --labels_file labels.txt --output embeddings/labels.pt

"""
