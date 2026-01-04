# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT with DRO support."""

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

# DRO constants
SPIKY_LOSS_FACTOR = 10

def _add_dro_args(p):
    """Add DRO hyperparameters to Megatron's parser."""
    g = p.add_argument_group('DRO μ update')
    g.add_argument('--use-dro', action='store_true', help='Enable DRO')
    g.add_argument('--dro-beta', type=float, default=0.9,
                   help='EMA decay β for per-expert losses')
    g.add_argument('--dro-eta', type=float, default=0.05,
                   help='η₀ used in η = η₀ / √E')
    g.add_argument('--dro-alpha', type=float, default=-1.0,
                   help='Alpha for soft-maximin DRO loss. Recommended range: [-2, -1]. Set to 1.0 to recover original DRO implementation.')
    g.add_argument('--log-dro-signals', action='store_true',
                   help='Log additional DRO signals for debugging.')
    
    # g.add_argument('--dro-gate-tanh-scale', type=float, default=8.0,
    #                help='Scale for tanh squashing of gate logits (e.g., 8.0). 0 to disable.')
    
    g.add_argument('--dro-router-grad-clip', type=float, default=0.5,
                   help='Clip router gradients at this value (e.g., 0.5). 0 to disable.')
    g.add_argument('--dro-mu-update-clip', type=float, default=0.0,
                   help='Clip absolute DRO μ increments; 0 disables clipping.')

    return p

# Global DRO state (initialized after args are available)
_dro_state = None


def _infer_num_moe_layers(args):
    """Infer number of MoE layers from --moe-layer-freq pattern."""
    if args.num_experts is None or args.num_experts == 0:
        return 0

    freq = args.moe_layer_freq
    if isinstance(freq, int):
        pattern = [1 if (i % freq == 0) else 0 for i in range(args.num_layers)]
    elif isinstance(freq, list):
        pattern = freq
    else:
        raise RuntimeError("Illegal --moe-layer-freq argument provided!")

    if len(pattern) != args.num_layers:
        raise ValueError(
            f"moe_layer_freq pattern length {len(pattern)} does not match num_layers {args.num_layers}"
        )

    return sum(pattern)


def _ensure_dro_state(num_moe_layers: int):
    """Ensure DRO state tensors match the expected per-layer shape."""
    global _dro_state

    args = get_args()
    E = getattr(args, 'num_experts', 0)
    if E <= 0:
        raise RuntimeError("DRO state requested but num_experts is not positive.")

    if num_moe_layers <= 0:
        # Fall back to inferred value from config if hooks haven't populated gate logits yet.
        num_moe_layers = _infer_num_moe_layers(args)
    if num_moe_layers <= 0:
        raise RuntimeError("No MoE layers detected for DRO state initialization.")

    beta = getattr(args, 'dro_beta', 0.9)
    eta_base = getattr(args, 'dro_eta', 0.05)
    eta = eta_base / (E ** 0.5) if E > 0 else eta_base
    device = torch.cuda.current_device()

    needs_reinit = False
    if _dro_state is None:
        needs_reinit = True
    else:
        if (
            _dro_state['num_layers'] != num_moe_layers
            or _dro_state['num_experts'] != E
            or _dro_state['device'] != device
        ):
            needs_reinit = True

    if needs_reinit:
        ema_losses = torch.zeros((num_moe_layers, E), device=device)
        dro_mu = torch.full((num_moe_layers, E), 1.0 / E, device=device)
        _dro_state = {
            'num_layers': num_moe_layers,
            'num_experts': E,
            'beta': beta,
            'eta': eta,
            'device': device,
            'ema_losses': ema_losses,
            'mu': dro_mu,
            'last_update_iter': -1,
        }
    else:
        # Update hyperparameters in case they have changed mid-run.
        _dro_state['beta'] = beta
        _dro_state['eta'] = eta
        if _dro_state['ema_losses'].shape != (num_moe_layers, E):
            _dro_state['ema_losses'] = torch.zeros((num_moe_layers, E), device=device)
        if _dro_state['mu'].shape != (num_moe_layers, E):
            _dro_state['mu'] = torch.full((num_moe_layers, E), 1.0 / E, device=device)

    return _dro_state

def _sync_and_update(per_layer_losses: torch.Tensor, state: dict):
    """Synchronize and update DRO statistics across data parallel ranks."""
    args = get_args()

    dp = mpu.get_data_parallel_group()
    torch.distributed.all_reduce(per_layer_losses, op=torch.distributed.ReduceOp.SUM, group=dp)
    per_layer_losses /= torch.distributed.get_world_size(dp)

    ema_losses = state['ema_losses']
    ema_losses.mul_(state['beta']).add_(per_layer_losses, alpha=1.0 - state['beta'])

    iteration = args.iteration
    if iteration == state['last_update_iter']:
        return
    state['last_update_iter'] = iteration

    mu_update_clip = max(getattr(args, 'dro_mu_update_clip', 0.0), 0.0)
    delta = state['eta'] * ema_losses
    if mu_update_clip > 0.0:
        delta = torch.clamp(delta, min=-mu_update_clip, max=mu_update_clip)

    dro_mu = state['mu']
    dro_mu.add_(delta)
    dro_mu.copy_(dro_mu.softmax(dim=-1))


def _masked_probs(logits, k):
    """Compute masked probabilities for top-k routing."""
    topv, topi = logits.topk(k, dim=-1)
    probs = torch.zeros_like(logits)
    probs.scatter_(1, topi, torch.softmax(topv, dim=-1))
    return probs, topi, logits.logsumexp(dim=-1)



def compute_per_expert_loss(token_loss, gate_logits, k, dro_mu=None, log_signals=False):
    """
    Computes per-expert loss contribution.
    token_loss: [batch_size, seq_len]
    gate_logits: list of [seq_len, batch_size, num_experts]
    """
    args = get_args()
    E = args.num_experts
    device = token_loss.device
    num_layers = len(gate_logits)
    if num_layers == 0:
        raise ValueError("gate_logits must contain at least one MoE layer")
    per_exp = torch.zeros((num_layers, E), device=device)

    # ---- 1. bring token_loss to shape [B·S, 1] ----
    token_loss = token_loss.reshape(-1, 1)

    # For logging signals
    signals = {}
    if log_signals:
        if dro_mu is None:
            raise ValueError("dro_mu must be provided when log_signals is True")
        signals_to_log = {
            'ess': [],
            'gate_logit_max': [],
            'topk_weight_max': [],
            'entropy': [],
        }

    # ---- 2. accumulate per-expert contributions ----
    for layer_idx, layer_logits in enumerate(gate_logits):  # Original shape: [S, B, E]
        # Permute to [B, S, E] then reshape to [B*S, E]
        S, B, E_ = layer_logits.shape
        layer_logits = layer_logits.permute(1, 0, 2).reshape(-1, E_)

        assert layer_logits.shape[0] == token_loss.shape[0], \
            "Shape mismatch after reshaping gate_logits"

        topv, topi = layer_logits.topk(k, dim=-1)  # [N, k]
        sel_probs = torch.softmax(topv, dim=-1)    # [N, k]

        contrib = token_loss * sel_probs            # [N, k]
        per_exp[layer_idx].scatter_add_(0, topi.reshape(-1), contrib.reshape(-1))

        if log_signals:
            with torch.no_grad():
                # ESS of token weights
                layer_mu = dro_mu[layer_idx]
                mu_selected = layer_mu[topi]
                token_weights = torch.sum(sel_probs * mu_selected, dim=-1)
                ess_layer = torch.sum(token_weights)**2 / (torch.sum(token_weights**2) + 1e-8)
                signals_to_log['ess'].append(ess_layer)

                # Router values
                signals_to_log['gate_logit_max'].append(layer_logits.max())
                signals_to_log['topk_weight_max'].append(sel_probs.max())
                
                # Per-token entropy H(E|X)
                log_probs_all = torch.log_softmax(layer_logits, dim=-1)
                probs_all = torch.exp(log_probs_all)
                entropy_layer = -torch.sum(probs_all * log_probs_all, dim=-1).mean()
                signals_to_log['entropy'].append(entropy_layer)

    if log_signals:
        # Average signals across layers
        for key in signals_to_log:
            signals[key] = torch.mean(torch.stack(signals_to_log[key])).item() if signals_to_log[key] else 0.0
        
    return per_exp, signals





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

    print_rank_0('building GPT model with DRO support ...')
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



    # Add DRO-specific attributes to the model for gate logits capture
    if args.num_experts > 0:
        model.gate_logits_buffer = []
        model._original_forward = model.forward
        
        clip_val = args.dro_router_grad_clip
        
        # Add hooks to capture gate logits from MoE routers
        for name, module in model.named_modules():
            if hasattr(module, 'router'):
                # Add hooks for router gradient clipping
                if clip_val > 0.0:
                    print_rank_0(f">>> Applying gradient clipping (value: {clip_val}) to router in '{name}'")
                    for p in module.router.parameters():
                        if p.requires_grad:
                            p.register_hook(lambda grad: torch.clamp(grad, -clip_val, clip_val))

                # Store original forward method
                original_forward = module.router.forward
                
                def create_hook(router, original_forward):
                    def hooked_forward(input_tensor):
                        # Apply input jitter
                        input_tensor = router.apply_input_jitter(input_tensor)
                        
                        # Get gate logits
                        logits = router.gating(input_tensor)
                        
                        # # Tanh-squash gate logits
                        # args = get_args()
                        # if args.dro_gate_tanh_scale > 0:
                        #     s = args.dro_gate_tanh_scale
                        #     logits = s * torch.tanh(logits / s)
                        
                        # Store gate logits for DRO computation
                        router.last_gate_logits = logits.detach()
                        
                        # Call the original routing method
                        scores, routing_map = router.routing(logits)
                        
                        return scores, routing_map
                    
                    return hooked_forward
                
                # Replace the router's forward method
                module.router.forward = create_hook(module.router, original_forward)
        
        def dro_forward(*args, **kwargs):
            """Enhanced forward pass that captures gate logits for DRO."""
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
        
        model.forward = dro_forward

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
    base_loss = loss.clone()

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
    
    # Apply DRO scaling if this is an MoE model
    if args.num_experts > 0:
        if gate_logits is None:
            raise RuntimeError("gate_logits not provided for DRO loss calculation")
            
        # Initialize DRO state
        state = _ensure_dro_state(len(gate_logits))
        dro_mu = state['mu']

        # Get the loss tensor for DRO computation
        token_losses = output_tensor

        # Compute per-expert losses using real gate logits
        k = args.moe_router_topk
        per_exp, dro_signals = compute_per_expert_loss(
            token_losses, gate_logits, k, dro_mu, getattr(args, 'log_dro_signals', False)
        )

        # Update DRO statistics
        dro_mu_before_update = dro_mu.clone() if getattr(args, 'log_dro_signals', False) else None
        with torch.no_grad():
            _sync_and_update(per_exp, state)

        # Compute DRO loss
        alpha = args.dro_alpha
        pi_g = dro_mu.detach()

        if alpha == 1.0:
            # Original DRO implementation: L_dro = sum(pi_g * l_g)
            # which is equivalent to scaling the total loss by sum(pi_g * (l_g / sum(l_g)))
            dro_loss = torch.sum(pi_g * per_exp)

        else:
            # Soft-maximin / power mean: L_dro = (sum(pi_g * l_g^alpha))^(1/alpha)
            # Add epsilon for numerical stability with negative alpha
            l_g = per_exp + 1e-8
            sum_term = torch.sum(pi_g * torch.pow(l_g, alpha))
            dro_loss = torch.pow(sum_term, 1.0 / alpha)

        # Replace original batch loss with DRO loss
        loss[0] = dro_loss

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    reporting_base_loss = base_loss.clone().detach()
    torch.distributed.all_reduce(reporting_base_loss, group=mpu.get_data_parallel_group())

    metrics = {
        'lm loss': (reporting_loss[0], reporting_loss[1]),
        'lm loss base': (reporting_base_loss[0], reporting_base_loss[1]),
    }
    if args.num_experts > 0 and getattr(args, 'log_dro_signals', False) and dro_mu_before_update is not None:
        # L1 norm of mu change (sum over layers and experts)
        delta_mu_l1 = torch.sum(torch.abs(dro_mu - dro_mu_before_update)).item()
        dro_signals['delta_mu_l1'] = delta_mu_l1
        
        # Format for logging
        for key, value in dro_signals.items():
            metrics[f'dro/{key}'] = (torch.tensor(value, device=loss.device), 1)

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

    # For DRO, we need to pass gate logits to the loss function
    if args.num_experts > 0:
        # The model might be wrapped in multiple layers (DistributedDataParallel, Float16Module, etc.)
        # Navigate through the wrappers to find the actual model
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
        extra_args_provider=_add_dro_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
