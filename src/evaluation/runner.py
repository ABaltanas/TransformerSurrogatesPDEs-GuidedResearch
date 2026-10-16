"""Batch execution runner for deterministic model evaluation."""

from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.interfaces import InferenceRunner
from src.training.interfaces import ModelWrapper


class BatchedInferenceRunner(InferenceRunner):
    """Executes deterministic batch inference across an unobserved test partition.

    Supports both ModelWrapper instances and standard PyTorch nn.Module architectures.
    """

    def predict(
        self,
        model: Any,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """Run batch inference on the test dataset under evaluation conditions.

        Parameters
        ----------
        model : Any
            The neural operator module (nn.Module) or ModelWrapper instance.
        dataloader : DataLoader
            DataLoader wrapping the test dataset partition.
        device : torch.device
            Hardware target device (e.g. cuda:0 or cpu).

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of concatenated predictions as NumPy arrays along the sample dimension.
        """
        # Obtain underlying torch module to toggle evaluation mode
        if isinstance(model, ModelWrapper):
            torch_module = model.get_torch_module()
        elif isinstance(model, nn.Module):
            torch_module = model
        else:
            raise TypeError(f"Expected nn.Module or ModelWrapper, got {type(model)}")

        torch_module.to(device)
        torch_module.eval()

        collected_outputs: Dict[str, List[np.ndarray]] = {}

        with torch.no_grad():
            for raw_batch in dataloader:
                batch = self._transfer_to_device(raw_batch, device)

                # Execute forward pass
                if isinstance(model, ModelWrapper):
                    preds = model.forward(batch)
                else:
                    if isinstance(batch, dict) and "inputs" in batch:
                        out = model(batch["inputs"])
                    else:
                        out = model(batch)
                    preds = out if isinstance(out, dict) else {"displacements": out}

                for k, v in preds.items():
                    if isinstance(v, torch.Tensor):
                        arr = v.detach().cpu().numpy()
                        if k not in collected_outputs:
                            collected_outputs[k] = []
                        collected_outputs[k].append(arr)

        # Concatenate batches along sample axis (axis 0)
        aggregated: Dict[str, np.ndarray] = {}
        for k, arr_list in collected_outputs.items():
            aggregated[k] = np.concatenate(arr_list, axis=0)

        return aggregated

    def _transfer_to_device(self, batch: Any, device: torch.device) -> Any:
        """Recursively send tensors within nested dictionary/list batches to target device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._transfer_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._transfer_to_device(v, device) for v in batch]
        return batch
