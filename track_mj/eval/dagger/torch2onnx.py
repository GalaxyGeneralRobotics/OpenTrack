import os
import json
import tyro
import torch
import numpy as np
import onnxruntime as rt
from dataclasses import dataclass

from track_mj.learning.models.dagger.policy_args import PolicyArgs
from track_mj.learning.models.dagger.policy import get_policy

MODEL_SIZE_THRESHOLD = 2 * 1024 * 1024 * 1024   # 2GB

@dataclass
class ExportArgs:
    ckpt_dir: str
    export_onnx: bool = True
    export_pt: bool = False

def get_latest_ckpt(dir):
    from pathlib import Path
    import re
    ckpts = [ckpt for ckpt in Path(dir).glob("*") if ckpt.name.endswith(".pth")]
    ckpts.sort(key=lambda x: int(re.search(r'(\d+)', x.stem).group(1)) if re.search(r'(\d+)', x.stem) else 0)
    return ckpts[-1] if ckpts else None

def convert_torch2onnx(
    ckpt_path: str,
    output_path: str,
    policy_args: PolicyArgs,
):

    rand_obs = {
        policy_args.policy_obs_key: torch.randn(1, policy_args.obs_dim).to(torch.float32),
        policy_args.policy_auxiliary_obs_key: torch.randn(1, policy_args.aux_obs_dim).to(torch.float32),
    }

    print(f"Initializing {policy_args.policy_type} policy...")
    model = get_policy(policy_args)
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        model.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        print("Checkpoint loaded.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    model.model.eval()
    model.model = model.model.to("cpu")
    
    export_model = model.model
    export_model.eval()
    export_model = export_model.to("cpu")

    if policy_args.policy_type != "mlp":
        raise NotImplementedError(f"Only mlp policy is supported, got: {policy_args.policy_type}")

    dummy_input = (rand_obs[policy_args.policy_obs_key],)
    input_names = ["obs"]
    output_names = ["continuous_actions"]

    with torch.no_grad():
        dummy_output = export_model(*dummy_input)

    dummy_output_np = [
        t.cpu().numpy() 
        for t in ([dummy_output] if isinstance(dummy_output, torch.Tensor) else dummy_output)
    ]

    assert len(input_names) == len(dummy_input) and len(output_names) == len(dummy_output_np), (
        "I/O name mismatch."
    )

    dynamic_shapes = ({0: torch.export.Dim("batch_size")},)


    torch.onnx.export(
        model = export_model,
        args = dummy_input,
        f = output_path,
        input_names = input_names,
        output_names = output_names,
        dynamic_shapes = dynamic_shapes,
        opset_version = 18,
        do_constant_folding=True,
        external_data = os.path.getsize(ckpt_path) > MODEL_SIZE_THRESHOLD,
    )

    sess = rt.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(output_names, {input_names[i]: dummy_input[i].cpu().numpy() for i in range(len(input_names))})
    
    for i in range(len(output_names)):
        print(f"[{i}]:{output_names[i]}  onnx_out_shape: {onnx_out[i].shape}. dummy_output_shape: {dummy_output_np[i].shape}")
    
    maes = {
        name: float(np.abs(o - t).mean())
        for name, o, t in zip(output_names, onnx_out, dummy_output_np, strict=False)
    }
    print(maes)

    return maes


def convert_torch2pt(
    ckpt_path: str,
    output_path: str,
    policy_args: PolicyArgs,
):
    """Convert PyTorch checkpoint to TorchScript (.pt) format."""
    
    rand_obs = {
        policy_args.policy_obs_key: torch.randn(1, policy_args.obs_dim).to(torch.float32),
        policy_args.policy_auxiliary_obs_key: torch.randn(1, policy_args.aux_obs_dim).to(torch.float32)
    }

    print(f"Initializing {policy_args.policy_type} policy...")
    model = get_policy(policy_args)
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        model.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        print("Checkpoint loaded.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    model.model.eval()
    model.model = model.model.to("cpu")
    
    if policy_args.policy_type != "mlp":
        raise NotImplementedError(f"Only mlp policy is supported, got: {policy_args.policy_type}")
    dummy_input = rand_obs[policy_args.policy_obs_key]

    with torch.no_grad():
        dummy_output = model.model(dummy_input)

    # Use torch.jit.script to convert the model
    print("Scripting model with TorchScript...")
    try:
        scripted = torch.jit.script(model.model)
    except Exception as e:
        print(f"torch.jit.script failed: {e}")
        print("Falling back to torch.jit.trace...")
        scripted = torch.jit.trace(model.model, dummy_input)
    
    scripted.save(output_path)
    print(f"TorchScript model saved to {output_path}")

    # Verify the exported model
    loaded_scripted = torch.jit.load(output_path)
    with torch.no_grad():
        scripted_output = loaded_scripted(dummy_input)
    
    dummy_output_np = dummy_output.cpu().numpy() if isinstance(dummy_output, torch.Tensor) else dummy_output[0].cpu().numpy()
    scripted_output_np = scripted_output.cpu().numpy() if isinstance(scripted_output, torch.Tensor) else scripted_output[0].cpu().numpy()
    
    mae = float(np.abs(scripted_output_np - dummy_output_np).mean())
    print(f"TorchScript MAE: {mae}")
    
    return {"continuous_actions": mae}


def main(export_args: ExportArgs):

    assert os.path.exists(export_args.ckpt_dir), f"Checkpoint directory does not exist: {export_args.ckpt_dir}"
    ckpt_dir = os.path.join(export_args.ckpt_dir, "checkpoints")
    with open(os.path.join(ckpt_dir, "config.json"), "r") as f:
        config_dict = json.load(f)
    policy_args = PolicyArgs.from_config_dict(config_dict["policy_config"]["policy_args"])

    latest_ckpt = get_latest_ckpt(ckpt_dir)
    assert latest_ckpt is not None, f"No checkpoint found in {ckpt_dir}"
    ckpt_path = str(latest_ckpt)
    
    if export_args.export_onnx:
        output_path_onnx = os.path.join(os.path.dirname(ckpt_path), "model.onnx")
        print(f"Exporting ONNX model to {output_path_onnx}...")
        convert_torch2onnx(
            ckpt_path=ckpt_path,
            output_path=output_path_onnx,
            policy_args=policy_args,
        )

    if export_args.export_pt:
        output_path_pt = os.path.join(os.path.dirname(ckpt_path), os.path.basename(ckpt_path).replace(".pth", ".pt"))
        print(f"Exporting TorchScript model to {output_path_pt}...")
        convert_torch2pt(
            ckpt_path=ckpt_path,
            output_path=output_path_pt,
            policy_args=policy_args,
        )

    if not export_args.export_onnx and not export_args.export_pt:
        print("Warning: No export format specified. Use --export-onnx and/or --export-pt.")

if __name__ == "__main__":
    main(tyro.cli(ExportArgs))