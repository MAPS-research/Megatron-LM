# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


stimer = StragglerDetector()

def _add_moe_signals_args(p):
    """Add MoE signals logging arguments."""
    g = p.add_argument_group('MoE Signals')
    g.add_argument('--log-moe-signals', action='store_true',
                   help='Log MoE router signals for debugging.')
    return p

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling
            )

    # Add hooks for logging MoE signals if enabled
    if args.num_experts > 0 and getattr(args, 'log_moe_signals', False):
        model.gate_logits_buffer = []
        model._original_forward = model.forward
        
        # Add hooks to capture gate logits from MoE routers
        for name, module in model.named_modules():
            if hasattr(module, 'router'):
                # Store original forward method
                original_forward = module.router.forward
                
                def create_hook(router, original_forward):
                    def hooked_forward(input_tensor):
                        # Apply input jitter
                        input_tensor = router.apply_input_jitter(input_tensor)
                        
                        # Get gate logits
                        logits = router.gating(input_tensor)
                        
                        # Store gate logits for logging
                        router.last_gate_logits = logits.detach()
                        
                        # Call the original routing method
                        scores, routing_map = router.routing(logits)
                        
                        return scores, routing_map
                    
                    return hooked_forward
                
                # Replace the router's forward method
                module.router.forward = create_hook(module.router, original_forward)
        
        def signals_forward(*args, **kwargs):
            """Enhanced forward pass that captures gate logits for logging."""
            # Clear previous gate logits
            model.gate_logits_buffer.clear()
            
            # Call original forward pass
            output = model._original_forward(*args, **kwargs)
            
            # Collect gate logits from all MoE layers
            for name, module in model.named_modules():
                if hasattr(module, 'router'):
                    # Access the router's gate logits directly
                    router = module.router
                    if hasattr(router, 'last_gate_logits') and router.last_gate_logits is not None:
                        model.gate_logits_buffer.append(router.last_gate_logits)
            
            return output
        
        model.forward = signals_forward

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


# Buffer for accumulating per-expert stats between checkpoint saves
_expert_stats_buffer = []
_expert_stats_call_count = 0


def _save_expert_stats_if_needed(stats_dict, force=False):
    """Accumulate per-expert stats and save at checkpoint intervals."""
    global _expert_stats_buffer, _expert_stats_call_count
    _expert_stats_buffer.append(stats_dict)
    _expert_stats_call_count += 1

    args = get_args()
    save_interval = getattr(args, 'save_interval', 1000)
    log_interval = getattr(args, 'log_interval', 5)
    # Save when we've accumulated enough micro-batches for one save_interval
    # Each iteration has (global_batch / (micro_batch * dp)) micro-batches calling this
    steps_per_save = max(1, save_interval // log_interval)
    if force or (_expert_stats_call_count % steps_per_save == 0):
        save_dir = os.path.join(os.environ.get('SSD_WEIGHTS', '.'), 'expert_stats')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'checkpoint_{_expert_stats_call_count}.pt')
        torch.save(_expert_stats_buffer, save_path)
        _expert_stats_buffer = []


def _compute_moe_signals(gate_logits, k, token_loss=None, loss_mask=None):
    """Computes MoE router signals and per-expert metrics.

    Full per-expert stats are accumulated and saved to disk at checkpoint intervals.
    Only summary stats are returned for wandb/console logging.

    Args:
        gate_logits: list of [S, B, E] tensors, one per MoE layer
        k: top-k routing value
        token_loss: optional [B, S] per-token loss tensor for per-expert loss
        loss_mask: optional [B*S] mask for valid tokens
    """
    signals_to_log = {
        'gate_logit_max': [],
        'topk_weight_max': [],
        'entropy': [],
    }
    signals = {}
    num_layers = len(gate_logits)
    if num_layers == 0:
        return signals

    E = gate_logits[0].shape[2]

    # Per-layer per-expert accumulators
    per_expert_loss = torch.zeros(num_layers, E) if token_loss is not None else None
    per_expert_routing_freq = torch.zeros(num_layers, E)
    per_expert_routing_weight = torch.zeros(num_layers, E)

    if token_loss is not None:
        flat_loss = (token_loss.float().view(-1) * loss_mask).view(-1, 1)  # [N, 1]

    for layer_idx, layer_logits in enumerate(gate_logits):  # Original shape: [S, B, E]
        S, B, E_ = layer_logits.shape
        layer_logits = layer_logits.permute(1, 0, 2).reshape(-1, E_)
        N = layer_logits.shape[0]

        with torch.no_grad():
            topv, topi = layer_logits.topk(k, dim=-1)  # [N, k]
            sel_probs = torch.softmax(topv, dim=-1)     # [N, k]

            # Aggregate signals (existing)
            signals_to_log['gate_logit_max'].append(layer_logits.max())
            signals_to_log['topk_weight_max'].append(sel_probs.max())

            log_probs_all = torch.log_softmax(layer_logits, dim=-1)
            probs_all = torch.exp(log_probs_all)
            entropy_layer = -torch.sum(probs_all * log_probs_all, dim=-1).mean()
            signals_to_log['entropy'].append(entropy_layer)

            # Per-expert routing frequency: count tokens per expert
            freq = torch.zeros(E_, device=layer_logits.device)
            freq.scatter_add_(0, topi.reshape(-1), torch.ones_like(topi.reshape(-1), dtype=freq.dtype))
            per_expert_routing_freq[layer_idx] = (freq / N).cpu()

            # Per-expert routing weight: sum of softmax weights per expert
            weight = torch.zeros(E_, device=layer_logits.device)
            weight.scatter_add_(0, topi.reshape(-1), sel_probs.reshape(-1))
            per_expert_routing_weight[layer_idx] = (weight / N).cpu()

            # Per-expert loss
            if token_loss is not None:
                exp_loss = torch.zeros(E_, device=layer_logits.device)
                contrib = flat_loss * sel_probs  # [N, k]
                exp_loss.scatter_add_(0, topi.reshape(-1), contrib.reshape(-1))
                per_expert_loss[layer_idx] = exp_loss.cpu()

    # Average aggregate signals across layers
    for key in signals_to_log:
        signals[key] = torch.mean(torch.stack(signals_to_log[key])).item() if signals_to_log[key] else 0.0

    # --- Summary stats for wandb/console ---
    # Per-layer worst (lowest) expert routing probability
    for l in range(num_layers):
        signals[f'worst_expert_prob_layer_{l}'] = per_expert_routing_freq[l].min().item()
    # Average expert routing probability across all layers
    signals['avg_expert_prob'] = per_expert_routing_freq.mean().item()

    if token_loss is not None:
        # Per-layer worst (highest) expert loss
        for l in range(num_layers):
            signals[f'worst_expert_loss_layer_{l}'] = per_expert_loss[l].max().item()
        # Average expert loss across all layers
        signals['avg_expert_loss'] = per_expert_loss.mean().item()

    # --- Save full per-expert stats to disk ---
    stats_to_save = {
        'per_expert_routing_freq': per_expert_routing_freq,     # [num_layers, E]
        'per_expert_routing_weight': per_expert_routing_weight,  # [num_layers, E]
    }
    if token_loss is not None:
        stats_to_save['per_expert_loss'] = per_expert_loss       # [num_layers, E]
    _save_expert_stats_if_needed(stats_to_save)

    return signals


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, gate_logits=None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    metrics = {'lm loss': (reporting_loss[0], reporting_loss[1])}
    if args.num_experts > 0 and getattr(args, 'log_moe_signals', False) and gate_logits:
        moe_signals = _compute_moe_signals(
            gate_logits, args.moe_router_topk,
            token_loss=losses, loss_mask=loss_mask,
        )
        for key, value in moe_signals.items():
            metrics[f'moe/{key}'] = (torch.tensor(value, device=loss.device), 1)

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        metrics,
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    # For MoE signal logging, we need to pass gate logits to the loss function
    if args.num_experts > 0 and getattr(args, 'log_moe_signals', False):
        actual_model = model
        while hasattr(actual_model, 'module'):
            actual_model = actual_model.module
        
        gate_logits = actual_model.gate_logits_buffer
        return output_tensor, partial(loss_func, loss_mask, gate_logits=gate_logits)
    else:
        return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=_add_moe_signals_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
