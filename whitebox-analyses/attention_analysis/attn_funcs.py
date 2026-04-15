import os
import sys
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from pytorch_models import analyze_text
from pytorch_models.model_config import model2layers_heads
from .tokenizer_funcs import get_raw_tokens
from tqdm import tqdm

from scipy import stats


def get_attention_matrix(text: str, model_name: str, layer: int, head: int, device_map: str = "auto") -> np.ndarray:
    """
    Get the attention matrix for a specific layer and head for given text.
    Note: This doesn't cache raw matrices as they can be very large.

    Args:
        text: Input text to analyze
        model_name: Name of the model (e.g., "qwen-14b", "llama8-base")
        layer: Layer index
        head: Head index
        device_map: Device mapping for model loading (default "auto", can be "cpu")

    Returns:
        Attention matrix as numpy array
    """
    result = analyze_text(
        text=text,
        model_name=model_name,
        verbose=False,
        return_logits=False,
        attn_layers=None,
        device_map=device_map,
    )

    if len(result["attention_weights"]) == 0:
        raise ValueError("No attention weights returned")

    matrix = result["attention_weights"][layer][0, head].numpy().astype(np.float32)
    return matrix


def generate_text_hash(text: str, sentences: Optional[List[str]] = None) -> str:
    """
    Generate a unique hash based on text content and optional chunk sentences.

    Args:
        text: The input text
        sentences: Optional list of sentences for chunking

    Returns:
        A hexadecimal hash string (first 16 characters of SHA256)
    """
    if sentences:
        content = text + "|||" + "|||".join(sentences)
    else:
        content = text

    hash_obj = hashlib.sha256(content.encode("utf-8"))
    return hash_obj.hexdigest()[:16]


def get_cache_path(
    cache_dir: Union[str, Path],
    text_id: str,
    model_name: str,
    layer: Union[int, List[int]],
    head: int,
    suffix: str = "",
) -> str:
    """
    Generate cache file path for a specific attention matrix.

    Args:
        cache_dir: Base cache directory
        text_id: Unique text identifier
        model_name: Model name
        layer: Layer index
        head: Head index
        suffix: Additional suffix for filename

    Returns:
        Path to cache file
    """
    if isinstance(layer, list):
        layer = "_".join(map(str, layer))
    filename = f"{layer}_{head}{suffix}.npy"

    Path(os.path.join(cache_dir, model_name, text_id, filename)).parent.mkdir(parents=True, exist_ok=True)
    return os.path.join(cache_dir, model_name, text_id, filename)


def get_sentence_token_boundaries(text: str, sentences: List[str], model_name: str) -> List[Tuple[int, int]]:
    """
    Get exact token boundaries for sentences within the full text.
    This accounts for tokenization effects where tokens may be different
    when sentences are tokenized together vs separately.

    Args:
        text: Full text containing all sentences
        sentences: List of sentence strings
        model_name: Model name for tokenizer

    Returns:
        List of (start, end) token positions for each sentence
    """
    if not sentences:
        return None

    import re

    def normalize_spaces(s: str) -> str:
        """Replace various Unicode spaces with regular space."""
        return re.sub(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]", " ", s)

    char_positions = []
    search_start = 0

    text_normalized = normalize_spaces(text)

    for sentence in sentences:
        sentence_normalized = normalize_spaces(sentence)

        norm_pos = text_normalized.find(sentence_normalized, search_start)
        if norm_pos == -1:
            sentence_stripped = sentence_normalized.strip()
            norm_pos = text_normalized.find(sentence_stripped, search_start)
            if norm_pos == -1:
                raise ValueError(f"Sentence not found in text: {sentence}")
            norm_end = norm_pos + len(sentence_stripped)
        else:
            norm_end = norm_pos + len(sentence_normalized)

        original_pos = 0
        normalized_count = 0
        actual_start = -1
        actual_end = -1

        for i, char in enumerate(text):
            if normalized_count == norm_pos and actual_start == -1:
                actual_start = i
            if normalized_count == norm_end:
                actual_end = i
                break
            if normalize_spaces(char) == " " or char == text_normalized[normalized_count]:
                normalized_count += 1

        if actual_end == -1 and normalized_count == norm_end:
            actual_end = len(text)

        char_positions.append((actual_start, actual_end))
        search_start = norm_end

    token_boundaries = []

    for char_start, char_end in char_positions:
        if char_start > 0:
            tokens_to_start = len(get_raw_tokens(text[:char_start], model_name))
        else:
            tokens_to_start = 0

        tokens_to_end = len(get_raw_tokens(text[:char_end], model_name))

        token_boundaries.append((tokens_to_start, tokens_to_end))

    return token_boundaries


def _compute_averaged_matrix(matrix: np.ndarray, sentence_boundaries: List[Tuple[int, int]]) -> np.ndarray:
    """
    Helper function to compute averaged matrix from raw matrix and boundaries.

    Args:
        matrix: Raw attention matrix
        sentence_boundaries: List of (start, end) tuples for each sentence

    Returns:
        Averaged matrix where each cell (i,j) is the average attention
        from sentence i to sentence j
    """
    if sentence_boundaries is None:
        return matrix

    n = len(sentence_boundaries)
    result = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        row_start, row_end = sentence_boundaries[i]
        row_start = min(row_start, matrix.shape[0] - 1)
        row_end = min(row_end, matrix.shape[0] - 1)

        if row_start >= row_end:
            continue

        for j in range(n):
            col_start, col_end = sentence_boundaries[j]
            col_start = min(col_start, matrix.shape[1] - 1)
            col_end = min(col_end, matrix.shape[1] - 1)

            if col_start >= col_end:
                continue

            region = matrix[row_start:row_end, col_start:col_end]
            if region.size > 0:
                result[i, j] = np.mean(region)

    return result


def _compute_averaged_matrix_fast(matrix: np.ndarray, sentence_boundaries: List[Tuple[int, int]]) -> np.ndarray:
    """
    Vectorized version of _compute_averaged_matrix using 2D cumulative sum.
    Complexity: O(seq²) for cumsum + O(n²) for lookups, vs O(n² × block²) for the loop version.
    """
    n = len(sentence_boundaries)
    H, W = matrix.shape

    row_starts = np.array([min(s, H - 1) for s, _ in sentence_boundaries])
    row_ends   = np.array([min(e, H - 1) for _, e in sentence_boundaries])
    col_starts = np.array([min(s, W - 1) for s, _ in sentence_boundaries])
    col_ends   = np.array([min(e, W - 1) for _, e in sentence_boundaries])

    # padded cumsum: cumsum[i, j] = sum(matrix[:i, :j])
    padded = np.zeros((H + 1, W + 1), dtype=np.float64)
    padded[1:, 1:] = matrix
    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    rs = row_starts[:, None]  # (n, 1)
    re = row_ends[:, None]    # (n, 1)
    cs = col_starts[None, :]  # (1, n)
    ce = col_ends[None, :]    # (1, n)

    totals = cumsum[re, ce] - cumsum[rs, ce] - cumsum[re, cs] + cumsum[rs, cs]
    counts = (re - rs) * (ce - cs)

    valid = (re > rs) & (ce > cs) & (counts > 0)
    result = np.where(valid, totals / np.where(counts > 0, counts, 1), 0.0)
    return result.astype(np.float32)


def compute_all_attention_matrices(
    text: str,
    model_name: str,
    sentences: Optional[List[str]],
    cache_dir: str = "avg_matrices",
    text_id: Optional[str] = None,
    device_map: str = "auto",
    force_recompute: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Compute attention matrices for all layers and heads at once.
    Uses a per-layer hook callback so only one layer is in RAM at a time (~2GB peak vs ~96GB).

    Returns:
        True if successful, False otherwise
    """
    n_layers, n_heads = model2layers_heads(model_name)

    if cache_dir and not text_id:
        text_id = generate_text_hash(text, sentences)

    if cache_dir and text_id and not force_recompute:
        all_exist = True
        for layer in range(n_layers):
            for head in range(n_heads):
                cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
                if not os.path.exists(cache_path):
                    all_exist = False
                    break
            if not all_exist:
                break

        if all_exist:
            if verbose:
                print(f"All matrices for {text_id} already exist in cache")
            return True

    if verbose:
        print(f"Computing attention matrices for {text_id}...")

    tokens = get_raw_tokens(text, model_name)
    if os.name == "nt" and len(tokens) > 3000:
        device_map = "cpu"

    sentence_boundaries = None
    if sentences:
        sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)

    # Callback invoked per layer during the forward pass.
    # Keeps computation on GPU: only the tiny (n_sentences × n_sentences) result is
    # transferred to CPU, not the full (n_heads × seq × seq) attention tensor.
    layers_done = [0]
    n_sents = len(sentence_boundaries) if sentence_boundaries else 0

    def cache_layer_callback(layer_idx, attn_tensor):
        import torch
        # attn_tensor: (1, n_heads, seq, seq) on GPU, float16
        device = attn_tensor.device
        seq_len = attn_tensor.shape[-1]

        with torch.no_grad():
            if sentence_boundaries is None:
                # No sentence chunking — must transfer full matrix to CPU
                attn_cpu = attn_tensor[0].cpu().float().numpy()
                for head in range(n_heads):
                    if cache_dir and text_id:
                        path = get_cache_path(cache_dir, text_id, model_name, layer_idx, head)
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        np.save(path, attn_cpu[head])
                return

            # Clamp sentence boundaries to this sequence's actual length
            row_s = torch.tensor([min(s, seq_len - 1) for s, _ in sentence_boundaries],
                                 dtype=torch.long, device=device)
            row_e = torch.tensor([min(e, seq_len - 1) for _, e in sentence_boundaries],
                                 dtype=torch.long, device=device)
            col_s = torch.tensor([min(s, seq_len - 1) for s, _ in sentence_boundaries],
                                 dtype=torch.long, device=device)
            col_e = torch.tensor([min(e, seq_len - 1) for _, e in sentence_boundaries],
                                 dtype=torch.long, device=device)

            # (n, n) block index arrays — tiny, built once per layer
            re_nn = row_e.unsqueeze(1).expand(n_sents, n_sents)
            rs_nn = row_s.unsqueeze(1).expand(n_sents, n_sents)
            ce_nn = col_e.unsqueeze(0).expand(n_sents, n_sents)
            cs_nn = col_s.unsqueeze(0).expand(n_sents, n_sents)

            counts = ((re_nn - rs_nn) * (ce_nn - cs_nn)).float()
            valid  = (re_nn > rs_nn) & (ce_nn > cs_nn) & (counts > 0)

            # Flatten indices for gather (avoids mixed basic/advanced indexing)
            stride    = seq_len + 1
            lin_re_ce = (re_nn * stride + ce_nn).reshape(-1)
            lin_rs_ce = (rs_nn * stride + ce_nn).reshape(-1)
            lin_re_cs = (re_nn * stride + cs_nn).reshape(-1)
            lin_rs_cs = (rs_nn * stride + cs_nn).reshape(-1)

            # Process heads in small batches to keep extra GPU memory under ~2 GB
            HEAD_BATCH = 8
            all_results = []

            for b0 in range(0, n_heads, HEAD_BATCH):
                b1   = min(b0 + HEAD_BATCH, n_heads)
                batch = attn_tensor[0, b0:b1].float()        # (bs, seq, seq)
                bs    = batch.shape[0]

                pad = torch.zeros(bs, seq_len + 1, seq_len + 1,
                                  dtype=torch.float32, device=device)
                pad[:, 1:, 1:] = batch
                del batch
                cumsum = pad.cumsum(dim=1).cumsum(dim=2)     # (bs, seq+1, seq+1)
                del pad

                flat   = cumsum.reshape(bs, -1)
                del cumsum
                totals = (flat[:, lin_re_ce] - flat[:, lin_rs_ce]
                          - flat[:, lin_re_cs] + flat[:, lin_rs_cs]).reshape(bs, n_sents, n_sents)
                del flat

                batch_result = torch.where(
                    valid.unsqueeze(0),
                    totals / counts.unsqueeze(0).clamp(min=1),
                    torch.zeros(1, dtype=torch.float32, device=device),
                )  # (bs, n_sents, n_sents)

                all_results.append(batch_result.cpu().numpy().astype(np.float32))
                del totals, batch_result

            result_np = np.concatenate(all_results, axis=0)  # (n_heads, n_sents, n_sents)

        if cache_dir and text_id:
            for head in range(n_heads):
                path = get_cache_path(cache_dir, text_id, model_name, layer_idx, head)
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                np.save(path, result_np[head])

        layers_done[0] += 1
        if verbose:
            print(f"\r  Cached layer {layers_done[0]}/{n_layers}", end="", flush=True)

    result = analyze_text(
        text,
        model_name=model_name,
        verbose=verbose,
        float32=(model_name == "qwen-15b"),
        attn_layers=None,
        return_logits=False,
        device_map=device_map,
        attn_hook_callback=cache_layer_callback,
    )

    if verbose:
        print()

    if layers_done[0] == n_layers:
        return True

    # Fallback: hook didn't fire (unusual), try reading from result directly
    if len(result["attention_weights"]) == 0:
        if verbose:
            print("No attention weights returned")
        return False

    for layer in tqdm(range(n_layers), desc="Saving avg. matrices"):
        for head in range(n_heads):
            matrix = result["attention_weights"][layer][0, head].numpy().astype(np.float32)
            if sentence_boundaries:
                matrix = _compute_averaged_matrix_fast(matrix, sentence_boundaries)
            if cache_dir and text_id:
                path = get_cache_path(cache_dir, text_id, model_name, layer, head)
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                np.save(path, matrix)

    return True


def get_avg_attention_matrix(
    text: str,
    model_name: str,
    layer: int,
    head: int,
    sentences: Optional[List[str]],
    device_map: str = "auto",
    cache_dir: Optional[str] = "attn_cache",
    text_id: Optional[str] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Get averaged attention matrix for a specific layer and head.

    Args:
        text: Input text to analyze
        model_name: Name of the model
        layer: Layer index
        head: Head index
        sentences: Optional list of sentences to chunk the text by
        device_map: Device mapping for model loading
        cache_dir: Directory to cache matrices (if None, no caching)
        text_id: Unique identifier for the text (auto-generated if not provided)
        force_recompute: Force recomputation even if cache exists

    Returns:
        Averaged attention matrix as numpy array
    """
    if cache_dir and not text_id:
        text_id = generate_text_hash(text, sentences)

    if cache_dir and text_id and not force_recompute:
        cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
        if os.path.exists(cache_path):
            return np.load(cache_path)

    if cache_dir and text_id:
        success = compute_all_attention_matrices(
            text=text,
            model_name=model_name,
            sentences=sentences,
            cache_dir=cache_dir,
            text_id=text_id,
            device_map=device_map,
            force_recompute=force_recompute,
            verbose=False,
        )
        if success:
            cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
            if os.path.exists(cache_path):
                return np.load(cache_path)
    # print('end')
    # quit()

    matrix = get_attention_matrix(text, model_name, layer, head, device_map)

    if sentences is None:
        return matrix

    sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)

    result = _compute_averaged_matrix(matrix, sentence_boundaries)

    if cache_dir and text_id:
        cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)

    return result


def get_vertical_scores(
    avg_mat: np.ndarray,
    proximity_ignore: int = 20,
    control_depth: bool = True,
    score_type: str = "mean",
) -> np.ndarray:
    """
    Calculate vertical attention scores from an averaged attention matrix.

    Args:
        avg_mat: Averaged attention matrix
        proximity_ignore: Number of nearby tokens to ignore (default 20)
        control_depth: Whether to multiply by depth/position (default True)
        score_type: How to aggregate scores - "mean" or "median" (default "mean")

    Returns:
        Array of vertical scores
    """

    # Clean the matrix - set upper triangle to NaN
    n = avg_mat.shape[0]
    trius = np.triu_indices_from(avg_mat, k=1)

    avg_mat = avg_mat.copy()
    avg_mat[trius] = np.nan

    trils = np.triu_indices_from(avg_mat, k=-proximity_ignore + 1)  # has no effect if not subtracting avg
    avg_mat[trils] = np.nan

    if control_depth:
        per_row = np.sum(~np.isnan(avg_mat), axis=1)
        avg_mat = stats.rankdata(avg_mat, axis=1, nan_policy="omit") / per_row[:, None]

    n = avg_mat.shape[-1]
    vert_scores = []

    for i in range(n):
        vert_lines = avg_mat[i + proximity_ignore:, i]

        if score_type == "mean":
            if len(vert_lines) == 0:
                # prevents "RuntimeWarning: Mean of empty slice"
                vert_score = np.nan
            else:
                vert_score = np.nanmean(vert_lines)
        elif score_type == "median":
            if len(vert_lines) == 0:
                vert_score = np.nan
            else:
                vert_score = np.nanmedian(vert_lines)
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        vert_scores.append(vert_score)

    return np.array(vert_scores)


def get_attention_to_step(
    text: str,
    model_name: str,
    layer: int,
    head: int,
    step_idx: int,
    sentences: List[str],
    device_map: str = "auto",
    cache_dir: Optional[str] = "attn_cache",
) -> np.ndarray:
    """
    Get attention from all tokens to a specific step/sentence.

    Args:
        text: Input text to analyze
        model_name: Name of the model
        layer: Layer index
        head: Head index
        step_idx: Index of the target step/sentence
        sentences: List of sentences for chunking
        device_map: Device mapping for model loading
        cache_dir: Directory to cache matrices

    Returns:
        Array of attention weights to the target step
    """
    avg_matrix = get_avg_attention_matrix(
        text,
        model_name,
        layer,
        head,
        sentences,
        device_map,
        cache_dir=cache_dir,
    )

    return avg_matrix[:, step_idx]
