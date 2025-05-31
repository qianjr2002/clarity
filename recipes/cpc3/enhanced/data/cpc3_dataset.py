import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
import torchaudio


class CPC3Dataset(Dataset):
    """Dataset for CPC3 hearing aid signals with metadata integration."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        target_sample_rate: int = 32000,
        use_demo_data: bool = False,
        force_mono: bool = False
    ):
        """
        Args:
            data_root: Path to clarity_CPC3_data directory
            split: "train" or "dev"
            target_sample_rate: Target sample rate for resampling (default 32kHz)
            use_demo_data: Whether to use demo data (clarity_demo_data)
            force_mono: Convert stereo to mono by averaging channels
        """
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.sample_rate = target_sample_rate
        self.dataset_type = "clarity_demo_data" if use_demo_data else "clarity_data"
        self.force_mono = force_mono
        
        if self.split not in ["train", "dev"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'dev'.")
        
        self.metadata = self._load_metadata()
        self.listener_info = self._load_listener_info() if self.split == "train" else None
        self.valid_pairs = self._validate_pairs()

    def _load_metadata(self) -> List[Dict]:
        """Load metadata JSON."""
        metadata_path = (
            self.data_root / self.dataset_type / "metadata" / f"CPC3.{self.split}.json"
        )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            return json.load(f)

    def _load_listener_info(self) -> Dict[str, str]:
        """Load listener severity info (only for training set)."""
        listeners_path = self.data_root / self.dataset_type / "metadata" / "listeners.csv"
        if not listeners_path.exists():
            raise FileNotFoundError(f"Listener info not found: {listeners_path}")
        
        with open(listeners_path, "r") as f:
            reader = csv.DictReader(f)
            return {row["listener_id"]: row["severity"] for row in reader}

    def _validate_pairs(self) -> List[Tuple[Path, Path, Dict]]:
        """Check and collect valid (signal, reference, metadata) pairs."""
        signals_dir = self.data_root / self.dataset_type / self.split / "signals"
        refs_dir = self.data_root / self.dataset_type / self.split / "references"
        valid_pairs = []
        
        for item in self.metadata:
            if self.split == "train":
                signal_path = signals_dir / f"{item['signal']}.wav"
                scene_id = item['signal'].split("_")[2]
                ref_path = refs_dir / f"{item['signal'].split('_')[0]}_{scene_id}_ref.wav"
            else:  # dev
                signal_path = signals_dir / f"{item['signal']}.wav"
                ref_path = refs_dir / f"{item['signal']}_ref.wav"
            
            if signal_path.exists() and ref_path.exists():
                valid_pairs.append((signal_path, ref_path, item))
            else:
                print(f"[Warning] Missing files for: {item['signal']}")
        
        return valid_pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Dict:
        signal_path, ref_path, metadata = self.valid_pairs[idx]
        
        # Load audio
        signal, signal_sr = torchaudio.load(signal_path)
        reference, ref_sr = torchaudio.load(ref_path)
        
        # Force mono if needed
        if self.force_mono:
            signal = signal.mean(dim=0, keepdim=True)
            reference = reference.mean(dim=0, keepdim=True)

        # Resample if necessary
        if signal_sr != self.sample_rate:
            signal = torchaudio.functional.resample(signal, signal_sr, self.sample_rate)
        if ref_sr != self.sample_rate:
            reference = torchaudio.functional.resample(reference, ref_sr, self.sample_rate)

        # Extract metadata
        if self.split == "train":
            listener_id = metadata["signal"].split("_")[3]
            severity = self.listener_info.get(listener_id, "Unknown")
            correctness = metadata["correctness"] / 100.0
            response = metadata["response"]
        else:
            severity = metadata["hearing_loss"]
            correctness = 0.0  # Not available
            response = ""  # No response in dev
        
        return {
            "signal": signal.squeeze(0),
            "reference": reference.squeeze(0),
            "signal_name": metadata["signal"],
            "correctness": correctness,
            "severity": severity,
            "prompt": metadata["prompt"],
            "response": response,
            "n_words": metadata["n_words"]
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, List]:
        """Custom collate function for variable-length audio."""
        return {
            "signals": [item["signal"] for item in batch],
            "references": [item["reference"] for item in batch],
            "correctness": torch.tensor([item["correctness"] for item in batch]),
            "ids": [item["signal_name"] for item in batch],  # <- 新增这一行
            "metadata": [{
                "signal_name": item["signal_name"],
                "severity": item["severity"],
                "prompt": item["prompt"],
                "response": item["response"],
                "n_words": item["n_words"]
            } for item in batch],
        }
