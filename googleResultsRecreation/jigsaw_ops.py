import torch

def permute_and_concat(patch_embeddings, perms):
    """
    patch_embeddings: [B, 9, C, H, W]
    perms: [M, 9]

    Returns:
        [B*M, C*9, H, W]
    """
    B, P, C, H, W = patch_embeddings.shape
    M = perms.shape[0]

    out = []

    for m in range(M):
        perm = perms[m]                      # [9]
        permuted = patch_embeddings[:, perm] # [B, 9, C, H, W]

        # move patch dimension into channel
        permuted = permuted.reshape(B, P*C, H, W)

        out.append(permuted)

    out = torch.stack(out, dim=1)            # [B, M, C*9, H, W]
    out = out.reshape(B * M, P * C, H, W)

    return out 