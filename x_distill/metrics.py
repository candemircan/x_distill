import torch

# all metrics except the last one are adapted from https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """Compute the biased HSIC (the original CKA)"""
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def cka(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    kernel_metric: str = "ip",
    rbf_sigma: float = 1.0,
    unbiased: bool = False,
) -> float:
    """
    Computes the Centered Kernal Alignment (CKA) between two representation matrices. Taken from [here](https://github.com/minyoungg/platonic-rep).

    Args:
        feats_A (torch.Tensor): _description_
        feats_B (torch.Tensor): _description_
        kernel_metric (str, optional): _description_. Defaults to "ip".
        rbf_sigma (float, optional): _description_. Defaults to 1.0.
        unbiased (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        float: _description_
    """

    if kernel_metric == "ip":
        # Compute kernel matrices for the linear case
        K = torch.mm(feats_A, feats_A.T)
        L = torch.mm(feats_B, feats_B.T)
    elif kernel_metric == "rbf":
        # COMPUTES RBF KERNEL
        K = torch.exp(-(torch.cdist(feats_A, feats_A) ** 2) / (2 * rbf_sigma**2))
        L = torch.exp(-(torch.cdist(feats_B, feats_B) ** 2) / (2 * rbf_sigma**2))
    else:
        raise ValueError(f"Invalid kernel metric {kernel_metric}")

    # Compute HSIC values
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk = hsic_fn(K, K)
    hsic_ll = hsic_fn(L, L)
    hsic_kl = hsic_fn(K, L)

    cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return cka_value.item()


def mutual_knn(feats_A, feats_B, topk):
    """
    Computes the mutual KNN accuracy.

    Args:
        feats_A: A torch tensor of shape N x feat_dim
        feats_B: A torch tensor of shape N x feat_dim

    Returns:
        A float representing the mutual KNN accuracy
    """
    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)

    n = knn_A.shape[0]
    topk = knn_A.shape[1]

    # Create a range tensor for indexing
    range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

    # Create binary masks for knn_A and knn_B
    lvm_mask = torch.zeros(n, n, device=knn_A.device)
    llm_mask = torch.zeros(n, n, device=knn_A.device)

    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0

    acc = (lvm_mask * llm_mask).sum(dim=1) / topk

    return acc.mean().item()


def compute_knn_accuracy(knn):
    """
    Compute the accuracy of the nearest neighbors. Assumes index is the gt label.
    Args:
        knn: a torch tensor of shape N x topk
    Returns:
        acc: a float representing the accuracy
    """
    n = knn.shape[0]
    acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
    acc = acc.float().view(n, -1).max(dim=1).values.mean()
    return acc


def compute_nearest_neighbors(feats, topk=10):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn


def cosine_similarity(x1, x2):
    return torch.nn.functional.cosine_similarity(
        x1.unsqueeze(1), x2.unsqueeze(0), dim=-1
    )


def class_separation(
    X: torch.Tensor,  # Feature by observation representation matrix.
    classes: torch.Tensor,  # Class labels for each observation.
) -> float:  # The class separation of X
    """
    Compute the class separation $R^2$ as defined in [this paper](https://arxiv.org/abs/2010.16402)
    """

    d_within = 0
    d_total = 0
    unique_classes = torch.unique(classes)
    total_classes = len(unique_classes)

    # Compute the within class distance
    for current_class in unique_classes:
        class_examples = X[classes == current_class]
        total_examples = len(class_examples)
        if total_examples > 1:
            pairwise_distances = 1 - cosine_similarity(class_examples, class_examples)
            d_within += pairwise_distances.sum() / (total_classes * total_examples**2)

    # Compute the total distance
    for cls_1 in unique_classes:
        cls_1_examples = X[classes == cls_1]
        total_examples_1 = len(cls_1_examples)
        for cls_2 in unique_classes:
            cls_2_examples = X[classes == cls_2]
            total_examples_2 = len(cls_2_examples)
            pairwise_distances = 1 - cosine_similarity(cls_1_examples, cls_2_examples)
            d_total += pairwise_distances.sum() / (
                total_classes**2 * total_examples_1 * total_examples_2
            )

    return 1 - d_within / d_total
