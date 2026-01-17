"""
Model Validation Script
Verifies that trained models are producing valid, sensible outputs
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import json


def load_model(model_path: str, device: str = 'cuda') -> tuple:
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return checkpoint


def check_output_validity(outputs: Dict[str, torch.Tensor], test_name: str) -> Dict[str, Any]:
    """Check if model outputs are valid"""
    results = {
        'test': test_name,
        'passed': True,
        'issues': []
    }

    for key, tensor in outputs.items():
        if tensor is None:
            continue

        # Check for NaN
        if torch.isnan(tensor).any():
            results['passed'] = False
            results['issues'].append(f"{key}: Contains NaN values")

        # Check for Inf
        if torch.isinf(tensor).any():
            results['passed'] = False
            results['issues'].append(f"{key}: Contains Inf values")

        # Check if all values are the same (model not learning)
        if tensor.numel() > 1:
            if tensor.std() < 1e-7:
                results['passed'] = False
                results['issues'].append(f"{key}: All values nearly identical (std={tensor.std().item():.2e})")

    return results


def check_prediction_range(predictions: torch.Tensor, min_val: float = 1.0, max_val: float = 5.0) -> Dict[str, Any]:
    """Check if predictions are in expected rating range"""
    results = {
        'test': 'prediction_range',
        'passed': True,
        'issues': [],
        'stats': {}
    }

    pred_min = predictions.min().item()
    pred_max = predictions.max().item()
    pred_mean = predictions.mean().item()
    pred_std = predictions.std().item()

    results['stats'] = {
        'min': pred_min,
        'max': pred_max,
        'mean': pred_mean,
        'std': pred_std
    }

    # Check if predictions are reasonable (allow some margin)
    if pred_min < min_val - 2 or pred_max > max_val + 2:
        results['issues'].append(f"Predictions outside expected range: [{pred_min:.2f}, {pred_max:.2f}]")

    # Check if there's variance in predictions
    if pred_std < 0.01:
        results['passed'] = False
        results['issues'].append(f"Very low prediction variance (std={pred_std:.4f}) - model may not be learning")

    return results


def check_input_sensitivity(model: nn.Module, base_inputs: Dict[str, torch.Tensor],
                           device: str = 'cuda') -> Dict[str, Any]:
    """Check if model outputs change when inputs change"""
    results = {
        'test': 'input_sensitivity',
        'passed': True,
        'issues': [],
        'sensitivity_scores': {}
    }

    model.eval()
    with torch.no_grad():
        # Get base output
        base_output = model(**base_inputs)
        if isinstance(base_output, dict):
            base_pred = base_output.get('predictions', base_output.get('popularity_prediction', list(base_output.values())[0]))
        else:
            base_pred = base_output

        # Test sensitivity to each input
        for input_name, input_tensor in base_inputs.items():
            if input_tensor is None or not isinstance(input_tensor, torch.Tensor):
                continue

            # Create modified input
            modified_inputs = base_inputs.copy()
            if input_tensor.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
                # Add noise to float inputs
                modified_inputs[input_name] = input_tensor + torch.randn_like(input_tensor) * 0.1
            elif input_tensor.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                # Shift integer inputs
                modified_inputs[input_name] = torch.clamp(input_tensor + 1, min=0)
            else:
                continue

            # Get modified output
            modified_output = model(**modified_inputs)
            if isinstance(modified_output, dict):
                modified_pred = modified_output.get('predictions', modified_output.get('popularity_prediction', list(modified_output.values())[0]))
            else:
                modified_pred = modified_output

            # Calculate difference
            diff = (base_pred - modified_pred).abs().mean().item()
            results['sensitivity_scores'][input_name] = diff

            if diff < 1e-6:
                results['issues'].append(f"Model insensitive to {input_name} changes")

    # Check if model is sensitive to at least one input
    if all(score < 1e-6 for score in results['sensitivity_scores'].values()):
        results['passed'] = False
        results['issues'].append("Model appears insensitive to all inputs")

    return results


def check_gradient_flow(model: nn.Module, inputs: Dict[str, torch.Tensor],
                        device: str = 'cuda') -> Dict[str, Any]:
    """Check if gradients flow through the model properly"""
    results = {
        'test': 'gradient_flow',
        'passed': True,
        'issues': [],
        'layer_stats': {}
    }

    model.train()

    # Forward pass
    outputs = model(**inputs)
    if isinstance(outputs, dict):
        pred = outputs.get('predictions', outputs.get('popularity_prediction', list(outputs.values())[0]))
    else:
        pred = outputs

    # Create dummy target and compute loss
    target = torch.rand_like(pred.squeeze()) * 4 + 1  # Random ratings 1-5
    loss = nn.MSELoss()(pred.squeeze(), target)

    # Backward pass
    loss.backward()

    # Check gradients for each layer
    zero_grad_layers = []
    nan_grad_layers = []
    exploding_grad_layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()

            if np.isnan(grad_norm):
                nan_grad_layers.append(name)
            elif grad_norm == 0:
                zero_grad_layers.append(name)
            elif grad_norm > 100:
                exploding_grad_layers.append((name, grad_norm))

            results['layer_stats'][name] = {
                'grad_norm': grad_norm if not np.isnan(grad_norm) else 'NaN',
                'param_norm': param.norm().item()
            }

    if nan_grad_layers:
        results['passed'] = False
        results['issues'].append(f"NaN gradients in {len(nan_grad_layers)} layers: {nan_grad_layers[:3]}...")

    if len(zero_grad_layers) > len(list(model.parameters())) * 0.5:
        results['passed'] = False
        results['issues'].append(f"Zero gradients in {len(zero_grad_layers)} layers (>50% of model)")

    if exploding_grad_layers:
        results['issues'].append(f"Large gradients in {len(exploding_grad_layers)} layers")

    model.zero_grad()
    return results


def validate_temporal_attention_model(model_path: str, device: str = 'cuda') -> Dict[str, Any]:
    """Validate a temporal attention model"""
    from src.models.hybrid.sota_tv.models.temporal_attention import TemporalAttentionTVModel

    print(f"\n{'='*60}")
    print(f"Validating: {model_path}")
    print(f"{'='*60}")

    results = {'model_path': model_path, 'tests': []}

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        results['error'] = str(e)
        return results

    # Create model
    vocab_sizes = {'shows': 50000, 'genres': 50, 'networks': 100}
    model = TemporalAttentionTVModel(vocab_sizes=vocab_sizes, d_model=512)

    # Load weights
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Model weights loaded")
    except Exception as e:
        print(f"✗ Failed to load model weights: {e}")
        results['error'] = str(e)
        return results

    model = model.to(device)
    model.eval()

    # Create test inputs
    batch_size = 32
    seq_len = 20

    test_inputs = {
        'show_ids': torch.randint(1, 10000, (batch_size, seq_len), device=device),
        'timestamps': torch.randint(1609459200, 1704067200, (batch_size, seq_len), device=device).float(),  # 2021-2024
        'genre_ids': None,
        'network_ids': None,
        'mask': None
    }

    # Test 1: Basic forward pass
    print("\n[Test 1] Basic Forward Pass...")
    try:
        with torch.no_grad():
            outputs = model(**test_inputs)
        result = check_output_validity(outputs, 'forward_pass')
        results['tests'].append(result)
        if result['passed']:
            print(f"  ✓ Forward pass produces valid outputs")
        else:
            print(f"  ✗ Issues: {result['issues']}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        results['tests'].append({'test': 'forward_pass', 'passed': False, 'error': str(e)})

    # Test 2: Prediction range
    print("\n[Test 2] Prediction Range Check...")
    try:
        with torch.no_grad():
            outputs = model(**test_inputs)
            pred = outputs.get('predictions', outputs.get('popularity_prediction'))
        result = check_prediction_range(pred)
        results['tests'].append(result)
        stats = result['stats']
        print(f"  Predictions: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        if result['passed']:
            print(f"  ✓ Predictions in reasonable range")
        else:
            print(f"  ✗ Issues: {result['issues']}")
    except Exception as e:
        print(f"  ✗ Prediction check failed: {e}")
        results['tests'].append({'test': 'prediction_range', 'passed': False, 'error': str(e)})

    # Test 3: Input sensitivity
    print("\n[Test 3] Input Sensitivity Check...")
    try:
        result = check_input_sensitivity(model, test_inputs, device)
        results['tests'].append(result)
        print(f"  Sensitivity scores: {result['sensitivity_scores']}")
        if result['passed']:
            print(f"  ✓ Model responds to input changes")
        else:
            print(f"  ✗ Issues: {result['issues']}")
    except Exception as e:
        print(f"  ✗ Sensitivity check failed: {e}")
        results['tests'].append({'test': 'input_sensitivity', 'passed': False, 'error': str(e)})

    # Test 4: Gradient flow
    print("\n[Test 4] Gradient Flow Check...")
    try:
        # Need fresh inputs with grad tracking
        test_inputs_grad = {
            'show_ids': torch.randint(1, 10000, (batch_size, seq_len), device=device),
            'timestamps': torch.randint(1609459200, 1704067200, (batch_size, seq_len), device=device).float(),
            'genre_ids': None,
            'network_ids': None,
            'mask': None
        }
        result = check_gradient_flow(model, test_inputs_grad, device)
        results['tests'].append(result)
        if result['passed']:
            print(f"  ✓ Gradients flow properly through model")
        else:
            print(f"  ✗ Issues: {result['issues']}")
    except Exception as e:
        print(f"  ✗ Gradient check failed: {e}")
        results['tests'].append({'test': 'gradient_flow', 'passed': False, 'error': str(e)})

    # Test 5: Consistency check (same input = same output)
    print("\n[Test 5] Output Consistency Check...")
    try:
        model.eval()
        with torch.no_grad():
            fixed_input = {
                'show_ids': torch.randint(1, 10000, (8, seq_len), device=device),
                'timestamps': torch.randint(1609459200, 1704067200, (8, seq_len), device=device).float(),
                'genre_ids': None,
                'network_ids': None,
                'mask': None
            }
            out1 = model(**fixed_input)
            out2 = model(**fixed_input)

            pred1 = out1.get('predictions', out1.get('popularity_prediction'))
            pred2 = out2.get('predictions', out2.get('popularity_prediction'))

            diff = (pred1 - pred2).abs().max().item()

            result = {
                'test': 'consistency',
                'passed': diff < 1e-5,
                'max_diff': diff,
                'issues': []
            }
            if not result['passed']:
                result['issues'].append(f"Inconsistent outputs for same input (diff={diff})")
            results['tests'].append(result)

            if result['passed']:
                print(f"  ✓ Model produces consistent outputs (max_diff={diff:.2e})")
            else:
                print(f"  ✗ Inconsistent outputs: {result['issues']}")
    except Exception as e:
        print(f"  ✗ Consistency check failed: {e}")
        results['tests'].append({'test': 'consistency', 'passed': False, 'error': str(e)})

    # Summary
    print(f"\n{'='*60}")
    passed = sum(1 for t in results['tests'] if t.get('passed', False))
    total = len(results['tests'])
    print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("✓ Model appears to be working correctly!")
    else:
        print("✗ Some tests failed - model may have issues")
        for test in results['tests']:
            if not test.get('passed', False):
                print(f"  - {test['test']}: {test.get('issues', test.get('error', 'Failed'))}")

    print(f"{'='*60}\n")

    return results


def validate_any_model(model_path: str, device: str = 'cuda') -> Dict[str, Any]:
    """Validate any model by checking checkpoint contents"""
    print(f"\n{'='*60}")
    print(f"Validating: {model_path}")
    print(f"{'='*60}")

    results = {'model_path': model_path, 'tests': []}

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        results['error'] = str(e)
        return results

    # Check checkpoint contents
    print(f"\nCheckpoint contents:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            if key == 'model_state_dict' or key == 'state_dict':
                state_dict = checkpoint[key]
                num_params = len(state_dict)
                total_elements = sum(p.numel() for p in state_dict.values())
                print(f"  - {key}: {num_params} tensors, {total_elements:,} parameters")

                # Check for NaN/Inf in weights
                nan_count = sum(1 for p in state_dict.values() if torch.isnan(p).any())
                inf_count = sum(1 for p in state_dict.values() if torch.isinf(p).any())

                if nan_count > 0:
                    print(f"    ✗ WARNING: {nan_count} tensors contain NaN values!")
                    results['tests'].append({'test': 'nan_weights', 'passed': False})
                else:
                    print(f"    ✓ No NaN values in weights")
                    results['tests'].append({'test': 'nan_weights', 'passed': True})

                if inf_count > 0:
                    print(f"    ✗ WARNING: {inf_count} tensors contain Inf values!")
                    results['tests'].append({'test': 'inf_weights', 'passed': False})
                else:
                    print(f"    ✓ No Inf values in weights")
                    results['tests'].append({'test': 'inf_weights', 'passed': True})
            elif key == 'epoch':
                print(f"  - {key}: {checkpoint[key]}")
            elif key == 'loss' or key == 'val_loss':
                loss_val = checkpoint[key]
                print(f"  - {key}: {loss_val:.6f}")
                if np.isnan(loss_val) or np.isinf(loss_val):
                    print(f"    ✗ WARNING: Loss is {loss_val}!")
            else:
                print(f"  - {key}: {type(checkpoint[key])}")

    return results


def find_model_checkpoints(directory: Path, patterns: list = None) -> list:
    """Recursively find all model checkpoint files"""
    if patterns is None:
        patterns = ['*.pt', '*.pth', '*.ckpt']

    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(directory.rglob(pattern))

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints


def validate_directory(directory: str, device: str = 'cuda', save_report: bool = True) -> Dict[str, Any]:
    """Validate all models in a directory recursively"""
    directory = Path(directory)

    if not directory.exists():
        print(f"Directory not found: {directory}")
        return {'error': 'Directory not found'}

    print(f"\n{'='*70}")
    print(f"SCANNING DIRECTORY: {directory}")
    print(f"{'='*70}")

    # Find all checkpoints
    checkpoints = find_model_checkpoints(directory)
    print(f"Found {len(checkpoints)} model checkpoint(s)\n")

    if not checkpoints:
        print("No model checkpoints found.")
        return {'checkpoints_found': 0, 'results': []}

    # Validate each checkpoint
    all_results = []
    summary = {
        'total': len(checkpoints),
        'passed': 0,
        'failed': 0,
        'errors': 0
    }

    for i, checkpoint_path in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] {checkpoint_path.relative_to(directory)}")
        print("-" * 50)

        # Determine model type from path
        path_str = str(checkpoint_path).lower()
        if 'temporal_attention' in path_str:
            model_type = 'temporal_attention'
        else:
            model_type = 'generic'

        try:
            if model_type == 'temporal_attention':
                result = validate_temporal_attention_model(str(checkpoint_path), device)
            else:
                result = validate_any_model(str(checkpoint_path), device)

            result['relative_path'] = str(checkpoint_path.relative_to(directory))
            all_results.append(result)

            # Update summary
            if 'error' in result:
                summary['errors'] += 1
            elif all(t.get('passed', False) for t in result.get('tests', [])):
                summary['passed'] += 1
            else:
                summary['failed'] += 1

        except Exception as e:
            print(f"  ✗ Error validating: {e}")
            all_results.append({
                'model_path': str(checkpoint_path),
                'relative_path': str(checkpoint_path.relative_to(directory)),
                'error': str(e)
            })
            summary['errors'] += 1

    # Print summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total checkpoints: {summary['total']}")
    print(f"  ✓ Passed: {summary['passed']}")
    print(f"  ✗ Failed: {summary['failed']}")
    print(f"  ⚠ Errors: {summary['errors']}")

    # List failed models
    if summary['failed'] > 0 or summary['errors'] > 0:
        print(f"\nProblematic models:")
        for result in all_results:
            if 'error' in result:
                print(f"  ⚠ {result.get('relative_path', result['model_path'])}: {result['error']}")
            elif not all(t.get('passed', False) for t in result.get('tests', [])):
                failed_tests = [t['test'] for t in result.get('tests', []) if not t.get('passed', False)]
                print(f"  ✗ {result.get('relative_path', result['model_path'])}: Failed {failed_tests}")

    print(f"{'='*70}\n")

    # Save report
    if save_report:
        report = {
            'directory': str(directory),
            'summary': summary,
            'results': all_results
        }
        report_path = directory / 'validation_report.json'
        with open(report_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, Path):
                    return str(obj)
                return obj
            json.dump(report, f, indent=2, default=convert)
        print(f"Full report saved to: {report_path}")

    return {'summary': summary, 'results': all_results}


def main():
    parser = argparse.ArgumentParser(description='Validate trained models')
    parser.add_argument('path', type=str, help='Path to model checkpoint or directory')
    parser.add_argument('--model-type', type=str, default='auto',
                       choices=['auto', 'temporal_attention', 'generic'],
                       help='Type of model to validate (ignored for directory scan)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--no-save', action='store_true', help='Do not save validation report')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    path = Path(args.path)

    # Check if path is a directory or file
    if path.is_dir():
        # Recursive directory validation
        validate_directory(args.path, args.device, save_report=not args.no_save)
    else:
        # Single file validation
        # Determine model type
        model_type = args.model_type
        if model_type == 'auto':
            if 'temporal_attention' in args.path.lower():
                model_type = 'temporal_attention'
            else:
                model_type = 'generic'

        # Run validation
        if model_type == 'temporal_attention':
            results = validate_temporal_attention_model(args.path, args.device)
        else:
            results = validate_any_model(args.path, args.device)

        # Save results
        if not args.no_save:
            results_path = Path(args.path).with_suffix('.validation.json')
            with open(results_path, 'w') as f:
                def convert(obj):
                    if isinstance(obj, (np.floating, np.integer)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, torch.Tensor):
                        return obj.tolist()
                    return obj

                json.dump(results, f, indent=2, default=convert)
            print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
