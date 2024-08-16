import torch
import triton
import triton.language as tl

from .kernels import group_triton_kernel, groupXtY_triton_kernel, scatter2scatter_triton_kernel


BLOCK_M = 128


@torch.compile
def padded_block_indices(sorted_experts_idxs: torch.Tensor, k: int, N_BLOCK_SIZE: int = BLOCK_M):
    expert_counts = torch.bincount(sorted_experts_idxs, minlength=k)
    padded_block_counts = ((expert_counts - 1) // N_BLOCK_SIZE) + 1
    padded_expert_block_end = padded_block_counts.cumsum(-1)
    expert_boundaries_end = expert_counts.cumsum(-1)
    expert_boundaries_start = expert_boundaries_end - expert_counts
    padded_expert_block_start = padded_expert_block_end - padded_block_counts

    block_idxs = torch.arange(
        padded_expert_block_end[-1], dtype=sorted_experts_idxs.dtype, device=sorted_experts_idxs.device
    ).unsqueeze(1)

    block_mask = (block_idxs < padded_expert_block_start) | (block_idxs >= padded_expert_block_end)
    expanded_block_idxs = N_BLOCK_SIZE * (block_idxs - padded_expert_block_start) + expert_boundaries_start
    expanded_block_idxs = expanded_block_idxs.masked_fill(block_mask, 0).sum(-1)

    return expanded_block_idxs, expert_boundaries_end


def scatter2scatter(
    X, W, sorted_expert_idxs, sorted_scattered_idxs, k, padded_block_idxs, x_grouped=False, y_grouped=False, out=None
):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k
    # Pre-kernel setup
    x_dim = X.size(-1)
    y_dim = W.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        O = torch.empty((L_scattered, y_dim), device=X.device, dtype=X.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        O = out

    def grid(META):
        grid_num = (padded_block_idxs.size(0) * triton.cdiv(META["N"], META["BLOCK_N"]),)
        return grid_num

    with torch.cuda.device(X.device):
        scatter2scatter_triton_kernel[grid](
            # X_ptr, stride_xm, stride_xk,
            X,
            X.stride(0),
            X.stride(1),
            # W_ptr, stride_we, stride_wk, stride_wn,
            W,
            W.stride(0),
            W.stride(1),
            W.stride(2),
            # Y_ptr, stride_ym, stride_yn,
            O,
            O.stride(0),
            O.stride(1),
            grouped_idx_ptr=sorted_scattered_idxs,
            expert_idxs_ptr=sorted_expert_idxs,
            block_start_idx_ptr=padded_block_idxs,
            FAN_OUT=k,
            M=X.size(0),
            K=X.size(1),
            N=O.size(1),
            E=W.size(0),
            BLOCK_M=BLOCK_M,
            ACC_TYPE=tl.float32,
            allow_tf32=True,
            x_grouped=x_grouped,
            y_grouped=y_grouped,
        )
        return O


def group_bwd_W(DY, X, expert_offsets, E):
    DWt = torch.zeros((E, DY.size(-1), X.size(-1)), device=DY.device, dtype=DY.dtype)
    DW = DWt.permute(0, 2, 1)

    def grid(META):
        grid = (
            E * triton.cdiv(META["K"], META["BLOCK_K"]),
            triton.cdiv(META["N"], META["BLOCK_N"]),
        )
        return grid

    with torch.cuda.device(DY.device):
        groupXtY_triton_kernel[grid](
            # DY_ptr, stride_dym, stride_dyk,
            DY,
            DY.stride(0),
            DY.stride(1),
            # X_ptr, stride_xm, stride_xn,
            X,
            X.stride(0),
            X.stride(1),
            # DW_ptr, stride_dwe, stride_dwk, stride_dwn,
            DW,
            DW.stride(0),
            DW.stride(1),
            DW.stride(2),
            # expert_offsets_ptr,
            expert_offsets,
            # K: tl.constexpr, N: tl.constexpr,
            N=DY.size(-1),
            K=X.size(-1),
            # ACC_TYPE: tl.constexpr,
            ACC_TYPE=tl.float32,
            allow_tf32=True,
        )
        return DW


def group(A, sorted_expert_idxs, coeff=None, fan_out=1, out=None):
    N = sorted_expert_idxs.size(0)
    K = A.size(1)
    assert A.size(0) * fan_out == N
    if out is not None:
        Y = out
    else:
        Y = torch.empty((N, K), dtype=A.dtype, device=A.device)
        # print("grp init:", Y.size())

    def grid(META):
        grid_num = (triton.cdiv(META["N"], META["BLOCK_N"]),)
        return grid_num

    with torch.cuda.device(A.device):
        group_triton_kernel[grid](
            # A_ptr, stride_an, stride_ai,
            A,
            A.stride(0),
            A.stride(1),
            coeff is not None,
            coeff,
            fan_out,
            # Y_ptr, stride_yn, stride_yk,
            Y,
            Y.stride(0),
            Y.stride(1),
            # grouped_idx_ptr,
            sorted_expert_idxs,
            # N: tl.constexpr, K: tl.constexpr,
            N,
            K,
        )
        return Y


class _ScatteredExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        output = scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=k,
            x_grouped=grouped_in,
            y_grouped=grouped_out,
        )

        if gates is None:
            output_expanded = None
        else:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = torch.bmm(gates[:, None, :], output_expanded).squeeze(1)

        ctx.save_for_backward(
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            output_expanded,
        )

        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_out):
        (
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            output_expanded,
        ) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out

        if gates is None:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None
        else:
            # calculate gates gradient
            d_gates = torch.bmm(output_expanded, grad_out[:, :, None]).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            # print("expanded and grouping")
            grouped_grad_out = output_expanded.flatten(0, 1)  # reuse expanded buffer later

        if grouped_out:
            grouped_grad_out = grad_out
        else:
            grouped_grad_out = group(
                grad_out, sorted_scattered_idxs, fan_out=gate_fan, coeff=gates_flat, out=grouped_grad_out
            )

        if grouped_in:
            grouped_x = x
            d_expanded_input = None
        else:
            grouped_x = group(x, sorted_scattered_idxs, fan_out=k)
            d_expanded_input = grouped_x

        d_weights = group_bwd_W(
            DY=grouped_grad_out, X=grouped_x, expert_offsets=expert_offsets, E=expert_weights.size(0)
        )

        d_expanded_input = scatter2scatter(
            X=grouped_grad_out,
            x_grouped=True,
            W=expert_weights.permute(0, 2, 1),
            padded_block_idxs=padded_block_idxs,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=1,
            y_grouped=grouped_in,
            out=d_expanded_input,  # Reuse grouped_x buffer
        )

        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)

        # print("backward end.")
        return (
            # x, expert_weights, k,
            d_input,
            d_weights,
            None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None,
            None,
            # padded_block_idxs, expert_offsets,
            None,
            None,
            # gates
            d_gates,
            None,
            None,
        )


def scattered_experts(
    inputs,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    padded_block_idxs,
    expert_offsets,
    gates=None,
    grouped_in=False,
    grouped_out=False,
):
    return _ScatteredExperts.apply(
        inputs,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates,
        grouped_in,
        grouped_out,
    )