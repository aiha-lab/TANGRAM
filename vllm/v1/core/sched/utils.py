# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

import torch

from vllm.v1.request import Request, RequestStatus


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def check_stop(
    request: Request, max_model_len: int, pooler_output: torch.Tensor | None = None
) -> bool:
    if request.pooling_params:
        if pooler_output is not None:
            # Multi-turn pooling requests advance the same way as generative.
            if request.advance_to_next_turn():
                request.status = RequestStatus.RUNNING
                return False
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

    sampling_params = request.sampling_params
    assert sampling_params is not None

    # ``min_tokens`` suppresses early EOS / stop-token termination but must
    # not override the per-turn ``max_tokens`` cap; otherwise a multi-turn
    # warm-up turn (max_tokens < min_tokens) can never reach
    # ``advance_to_next_turn`` and the engine hangs at 0 scheduled tokens.
    if (request.num_output_tokens < sampling_params.min_tokens
            and request.num_output_tokens < request.max_tokens):
        return False

    last_token_id = request.output_token_ids[-1]
    turn_stopped = False
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        turn_stopped = True
    elif last_token_id in (sampling_params.stop_token_ids or ()):
        turn_stopped = True
        request.stop_reason = last_token_id
    elif request.num_tokens >= max_model_len:
        # Model-length cap cannot be carried into another turn.
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    elif request.num_output_tokens >= request.max_tokens:
        # Per-turn cap reached — for multi-turn this is the advance trigger.
        turn_stopped = True

    if not turn_stopped:
        return False

    # ``advance_to_next_turn`` returns False for single-turn requests and
    # for the last turn of a multi-turn request — exactly when we want to
    # actually finish.
    if request.advance_to_next_turn():
        request.status = RequestStatus.RUNNING
        request.stop_reason = None
        return False
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
    elif last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
    else:
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return True
