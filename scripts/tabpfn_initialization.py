"""Construct the official TabPFN v2 network, omitting its trained neural state."""
import hashlib
from importlib.metadata import version

import torch
from tabpfn.base import RegressorModelSpecs
from tabpfn.model_loading import (
    ARCHITECTURES,
    _get_inference_config_from_checkpoint,
    get_loss_criterion,
    get_n_out,
)


def initialized_spec(weights, initialization_seed):
    """Mirror version 6.4.1's load_model up to the separate network-weight load.

    Upstream factory and loader (release tag v6.4.1):
    https://github.com/PriorLabs/TabPFN/blob/8848704423f56264d8590f9f3adc507f4015e176/src/tabpfn/model_loading.py
    """
    if version("tabpfn") != "6.4.1":
        raise RuntimeError("Initialization control was verified for tabpfn==6.4.1")
    # Caller first verifies the public checkpoint digest.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    architecture = ARCHITECTURES[checkpoint.get("architecture_name", "base")]
    config, _ = architecture.parse_config(checkpoint["config"])
    reference = checkpoint["state_dict"]
    criterion = get_loss_criterion(config)
    criterion_state = {k.removeprefix("criterion."): v for k, v in reference.items()
                       if k.startswith("criterion.")}
    criterion.load_state_dict(criterion_state, strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        network = architecture.get_architecture(
            config, n_out=get_n_out(config, criterion), cache_trainset_representation=True,
        )
    network.eval()
    # Compare every named neural parameter. Everything evaluated is from the
    # fresh constructor; only the separate output-distribution state was loaded.
    digest = hashlib.sha256()
    changed = []
    equal = []
    num_elements = 0
    for name, p in network.named_parameters():
        tensor = p.detach().cpu().contiguous()
        if name not in reference or tensor.shape != reference[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Fresh parameter does not match the expected layout: {name}")
        digest.update(name.encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
        (equal if torch.equal(tensor, reference[name]) else changed).append(name)
        num_elements += tensor.numel()
    if not changed or "decoder_dict.standard.0.weight" not in changed:
        raise ValueError("Fresh network matches the pretrained regression decoder")
    inference = _get_inference_config_from_checkpoint(checkpoint, criterion)
    return RegressorModelSpecs(network, config, inference, criterion), {
        "initialization_seed": initialization_seed, "neural_parameter_sha256": digest.hexdigest(),
        "parameter_elements": num_elements, "changed_parameter_tensors": len(changed),
        "identical_parameter_tensor_names": equal,
        "retained_checkpoint_criterion_keys": sorted(criterion_state),
        "retained": "Architecture config, v2 inference preprocessing, calibration labels, regression target-bin distribution, library's fixed seed-42 column-position vectors.",
        "factory_revision": "8848704423f56264d8590f9f3adc507f4015e176",
        "learned_neural_tensors_loaded_into_fresh_network": 0,
    }
