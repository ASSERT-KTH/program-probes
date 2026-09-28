"""Plot agent trajectories projected into 2D probe-derived subspaces.

Four projection modes (any combination, see --mode):

  per_probe  one figure per probe. A 2-class softmax's decision only depends on
             weight[1] - weight[0] (empirically ~rank 1: the two rows are nearly
             antiparallel), so there is no second *meaningful* axis inside the
             weight matrix itself -- a second SVD axis there would just be noise.
             Axis 1 = that exact decision direction (background shading, i.e.
             predicted P(class=1), is an exact function of this coordinate alone).
             Axis 2 = top PCA component of the residual activations after removing
             axis 1 -- the dominant direction the hidden state actually varies
             along that this probe doesn't encode. Axis 2 is orthogonal to axis 1
             by construction, so this is a real (if partly unsupervised) 2D space,
             not a degenerate one.
  combined   one figure. Axes = top-2 singular vectors of the (4, hidden_dim)
             matrix stacking all 4 probes' unit-normalized decision directions
             (weight[1] - weight[0]). Each probe's decision boundary is then only
             approximately recoverable in this plane (least-squares projection),
             shown as dashed contour lines rather than filled shading.
  pca        one figure. Unsupervised linear control: sklearn PCA fit on the raw
             activations themselves (no probe information). Same approximate
             per-probe contour overlay as `combined`.
  umap       one figure. Unsupervised *nonlinear* control (umap-learn). No
             boundary overlay -- a nonlinear embedding has no simple linear
             correspondence to a probe's decision direction, so there's nothing
             exact or even well-defined to draw there.

By default, each captured activation is a token 5-strided position within an
assistant turn, not a semantically meaningful checkpoint -- most steps share a
label carried forward from the last edit while their local token content (and
hence hidden state) varies a lot, which looks like noise. Pass
--after-edit-only to instead get exactly one point per real code edit, using
the same edit_step_index.pt used by src/probe.py's after_edit_only training
mode: rather than picking a single representative stride position and
discarding the rest, every captured state between consecutive edits is mean-
pooled into one point, INCLUDING the leading span before the first edit
("start") and the trailing span after the last edit ("end") -- every point on
the path is a mean over its span, none is a single raw snapshot. Mixing a
raw single-token point into an otherwise-pooled path would be misleading: a
raw point has far higher variance than a pooled mean purely as an estimator
artifact, so it could sit apart from the pooled cluster for reasons having
nothing to do with anything semantically happening at that point. This uses
all the hidden states -- smoother and less token-level-noisy than picking one
position -- while still keeping one point per edit (plus start/end). (For a
linear probe, a pooled point's exact P(positive) equals the mean of the
individual positions' P(positive), since logit is linear in H.)

Outputs both .pdf and .png per figure.

Example
-------
python scripts/plot_latent_trajectory.py \\
    --run-id laguna_xs2_full \\
    --instances astropy__astropy-12907 astropy__astropy-12907_run01 astropy__astropy-12907_run06 \\
    --mode per_probe combined pca umap \\
    --after-edit-only \\
    --output-dir paper/figures/trajectories/astropy-12907
"""

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from paper import style as _style
from run_paper_figures import PROBE_TO_PROPERTY, MODEL_LABELS, _model_key  # noqa: E402

ALL_PROBES = [
    "currently_compiles",
    "currently_correct",
    "currently_has_regressions",
    "currently_reduces_failing",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_probe_weights(results_dir: Path, probe_run_id: str, layer: int) -> dict[str, dict]:
    out = {}
    for probe in ALL_PROBES:
        p = results_dir / probe_run_id / probe / "weights.pt"
        d = torch.load(p, weights_only=False)
        bin0 = d[layer][0]
        out[probe] = {
            "weight": bin0["state_dict"]["weight"].float().numpy(),  # (2, hidden_dim)
            "bias": bin0["state_dict"]["bias"].float().numpy(),      # (2,)
            "mean": bin0["mean"].float().numpy(),                    # (hidden_dim,)
        }
    return out


def _load_edit_step_index(cache_dir: Path, activations_dir: Path, run_id: str) -> dict[str, list[int]]:
    dataset_name = Path(activations_dir).name  # "swebench" or "swebench_pro"
    idx_path = cache_dir / dataset_name / run_id / "edit_step_index.pt"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"--after-edit-only requires {idx_path}. Run build_edit_step_index.py first."
        )
    return torch.load(idx_path, weights_only=False)


def _load_trajectory(activations_dir: Path, run_id: str, instance_id: str, layer: int,
                      after_edit_only: bool = False,
                      edit_step_index: dict[str, list[int]] | None = None) -> dict:
    p = activations_dir / run_id / f"{instance_id}.pt"
    d = torch.load(p, weights_only=False)
    H = d["activations"][layer].float().numpy()  # (n_steps, hidden_dim)

    if after_edit_only:
        T = len(H)
        n_turns = d["n_turns"]
        # Mirrors src/build_cache.py's own stride-step -> turn-index mapping.
        step_to_turn = np.minimum((np.arange(T) * n_turns) // T, n_turns - 1)
        edit_turns = set((edit_step_index or {}).get(instance_id, []))
        # One point per edit, not one point per stride-5 token within that
        # edit's turn (a single verbose turn can span dozens of captured
        # positions). Rather than picking a single representative position and
        # discarding the rest, *pool* (mean) every captured state between
        # consecutive edits into one point -- uses all the hidden states
        # (smoother, less token-level noise) while keeping one point per edit.
        last_in_turn: dict[int, int] = {}
        for i in range(T):
            turn = step_to_turn[i]
            if turn in edit_turns:
                last_in_turn[turn] = i  # later i overwrites, so this ends up last
        boundaries = sorted(last_in_turn.values())
        if not boundaries:
            print(f"[after_edit_only] {instance_id}: no recorded edits, falling back to start/end only")
            H = H[[0, T - 1]]
        else:
            # Every point -- including the first ("start") and last ("end")
            # markers -- is a mean over its span, not a single raw snapshot.
            # A raw single-token state has far higher variance than a pooled
            # mean, so a raw point sitting apart from the pooled cluster would
            # just be a pooling-vs-snapshot estimator artifact, not a
            # meaningful jump; using the same estimator everywhere makes the
            # path's shape actually comparable point-to-point.
            pooled = []
            prev = -1  # start absorbs indices [0, boundaries[0]], not [1, ...]
            for b in boundaries:
                segment = H[prev + 1:b + 1]
                pooled.append(segment.mean(axis=0) if len(segment) else H[b])
                prev = b
            if prev != T - 1:
                tail = H[prev + 1:T]
                pooled.append(tail.mean(axis=0) if len(tail) else H[T - 1])
            H = np.stack(pooled)

    return {
        "instance_id": instance_id,
        "H": H,
        "outcome": d["outcome"],
    }


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def project(H: np.ndarray, basis: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """basis: (2, hidden_dim) orthonormal rows -> coords (n, 2)."""
    return (H - mean) @ basis.T


def probe_probs(H: np.ndarray, weight: np.ndarray, bias: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Exact P(class=1) for each row of H, straight from the probe's own
    (2, hidden_dim) weight matrix -- independent of whatever 2D basis is being
    plotted, since it's computed from the raw hidden state."""
    logits = (H - mean) @ weight.T + bias
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e[:, 1] / e.sum(axis=1)


def per_probe_basis(weight: np.ndarray, bias: np.ndarray, mean: np.ndarray,
                     H_pooled: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Build a genuine 2D basis for one probe: an *exact* decision axis plus a
    real (non-degenerate) second axis.

    A 2-class softmax's decision only depends on the difference of its two
    weight rows, so ``weight`` is effectively rank 1 (verified empirically: the
    two rows are ~antiparallel, second singular value ~6% of the first).
    Taking a second axis from ``weight`` itself would just be fitting noise.
    Instead: axis 1 = the exact decision direction ``weight[1] - weight[0]``
    (P(positive) is a pure function of this coordinate, no approximation);
    axis 2 = the top PCA component of the *residual* activations after
    removing axis 1 — i.e. the dominant direction the hidden state actually
    varies along that this probe doesn't care about. Axis 2 is automatically
    orthogonal to axis 1 (every residual row has zero component along axis 1
    by construction), so this is a real, non-degenerate 2D space.

    Returns (basis (2, hidden_dim), norm_d, bias_diff, explained_var_axis2)
    where the exact logit-difference at a projected coordinate ``(x, y)`` is
    ``x * norm_d + bias_diff`` (independent of y).
    """
    direction_raw = weight[1] - weight[0]
    norm_d = float(np.linalg.norm(direction_raw))
    direction_unit = direction_raw / norm_d
    bias_diff = float(bias[1] - bias[0])

    Hc = H_pooled - mean
    x = Hc @ direction_unit
    residual = Hc - np.outer(x, direction_unit)
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    basis_y = Vt[0]
    explained_y = float((S[0] ** 2) / (S ** 2).sum())

    basis = np.stack([direction_unit, basis_y])
    return basis, norm_d, bias_diff, explained_y


def combined_basis(weights: dict[str, dict], center: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    probes = ALL_PROBES
    directions = []
    for p in probes:
        d = weights[p]["weight"][1] - weights[p]["weight"][0]
        directions.append(d / np.linalg.norm(d))
    W = np.stack(directions)  # (4, hidden_dim)
    if center:
        W = W - W.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    basis = Vt[:2]
    explained = (S ** 2) / (S ** 2).sum()
    return basis, explained[:2], W, probes


# ---------------------------------------------------------------------------
# Plotting helpers (operate on precomputed 2D coords, not on H directly, so
# both linear (per_probe/combined/pca) and nonlinear (umap) modes can share
# them)
# ---------------------------------------------------------------------------

def _outcome_color(outcome: bool) -> str:
    return "#2E7D32" if outcome else "#C62828"  # green / red


def _time_cmap(model_key: str) -> mcolors.Colormap:
    """Light -> dark gradient in the model's own brand color (paper/style.py's
    COLORS: model identity = color family), used for the step-order gradient
    instead of a generic colormap like viridis -- keeps this figure visually
    consistent with the rest of the paper's model-color convention."""
    base = {
        "laguna_xs2_full":         _style.COLORS["laguna_verified"],
        "laguna_xs2_pro_full":     _style.COLORS["laguna_pro"],
        "qwen36_35b_a3b_full":     _style.COLORS["qwen_verified"],
        "qwen36_35b_a3b_pro_full": _style.COLORS["qwen_pro"],
    }.get(model_key, "#555555")
    return mcolors.LinearSegmentedColormap.from_list("time_cmap", ["#F0F0F0", base])


def _add_time_swatch(ax, cmap) -> None:
    """Low-key inset gradient bar explaining the dark/light time encoding,
    replacing the numeric colorbar (there's nothing to put numbers on --
    step order is relative, not an absolute quantity)."""
    inset = ax.inset_axes([0.03, 0.05, 0.24, 0.035])
    inset.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap=cmap)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)
    ax.annotate("earlier", xy=(0.03, 0.10), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=7, color="0.4")
    ax.annotate("later", xy=(0.27, 0.10), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color="0.4")


def _short_id(instance_id: str, max_len: int = 40) -> str:
    """SWE-bench Pro instance ids are long commit-hash-suffixed strings that
    would otherwise force matplotlib's tight_layout to squash the whole figure
    down to a thin strip to fit the title on one line."""
    return instance_id if len(instance_id) <= max_len else instance_id[:max_len - 1] + "…"


def _plot_trajectory(ax, coords: np.ndarray, outcome: bool, start_label: str | None,
                      outcome_label: str | None, scale: float = 1.0,
                      color_values: np.ndarray | None = None, cmap: str = "viridis",
                      vmin: float = 0.0, vmax: float = 1.0):
    """color_values: per-point scalar to color the path by (defaults to step
    order 0..1 if None); e.g. a probe's exact P(positive) at that point.

    Step order is always used for marker *size* (small -> large), not just
    color, so the direction of travel stays visible even when color is
    reassigned to something else (e.g. a probe probability) instead of time."""
    t = np.linspace(0, 1, len(coords))
    c = color_values if color_values is not None else t
    sizes = (6 + 10 * t) * scale if color_values is not None else 8 * scale
    ax.plot(coords[:, 0], coords[:, 1], color="0.6", linewidth=0.8 * scale, zorder=2, alpha=0.85)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                     s=sizes, zorder=3, linewidths=0)
    ax.scatter(*coords[0], marker="o", s=40 * scale, facecolor="white",
                edgecolor="black", zorder=4, linewidth=1.0, label=start_label)
    ax.scatter(*coords[-1], marker="*", s=120 * scale, facecolor=_outcome_color(outcome),
                edgecolor="black", zorder=5, linewidth=0.6, label=outcome_label)
    return sc


def _plot_trajectories(ax, trajectories: list[dict], coords_list: list[np.ndarray],
                        color_values_list: list[np.ndarray] | None = None,
                        cmap: str = "viridis"):
    """Overlay all trajectories on one axes (--overlay mode). One legend entry
    per outcome (resolved/unresolved), not per trajectory — with many reruns of
    the same instance a per-trajectory legend just covers the plot."""
    seen_outcomes: set[bool] = set()
    sc = None
    for i, (traj, coords) in enumerate(zip(trajectories, coords_list)):
        label = None
        if traj["outcome"] not in seen_outcomes:
            seen_outcomes.add(traj["outcome"])
            label = "resolved" if traj["outcome"] else "unresolved"
        cv = color_values_list[i] if color_values_list is not None else None
        sc = _plot_trajectory(ax, coords, traj["outcome"], start_label=None,
                               outcome_label=label, color_values=cv, cmap=cmap)
    return sc


def _plot_single_trajectory(ax, traj: dict, coords: np.ndarray,
                             color_values: np.ndarray | None = None, cmap: str = "viridis"):
    """Draw exactly one trajectory (default, one-figure-per-instance mode)."""
    outcome_label = "resolved" if traj["outcome"] else "unresolved"
    return _plot_trajectory(ax, coords, traj["outcome"], start_label="start",
                             outcome_label=outcome_label, scale=1.3,
                             color_values=color_values, cmap=cmap)


def _grid_for_coords(coords_list: list[np.ndarray], pad: float = 0.15, n: int = 200):
    """Percentile-based bounds (not min/max): LLM hidden states routinely have a
    handful of extreme-norm outlier tokens ('attention sink'-style activations)
    that would otherwise stretch the axes and squash the rest of the trajectory
    down to a sliver."""
    all_coords = np.concatenate(coords_list, axis=0)
    xmin, ymin = np.percentile(all_coords, 1, axis=0)
    xmax, ymax = np.percentile(all_coords, 99, axis=0)
    dx, dy = (xmax - xmin) * pad, (ymax - ymin) * pad
    xs = np.linspace(xmin - dx, xmax + dx, n)
    ys = np.linspace(ymin - dy, ymax + dy, n)
    return xs, ys, (xmin - dx, xmax + dx, ymin - dy, ymax + dy)


def _finalize_legend(ax, paper: bool = False) -> None:
    """Add a proxy 'start' entry (if not already present) and place the legend.

    Default: outside the axes (good while browsing many exploratory figures).
    paper=True: inside the axes in a data-sparse corner instead, so the right
    margin is free for just the colorbar (outside placement collided with it)."""
    handles, labels = ax.get_legend_handles_labels()
    if "start" not in labels:
        start_proxy = plt.Line2D([], [], marker="o", markersize=6, markerfacecolor="white",
                                  markeredgecolor="black", linestyle="")
        handles.append(start_proxy)
        labels.append("start")
    if paper:
        ax.legend(handles, labels, loc="best", fontsize=8, framealpha=0.9,
                   handletextpad=0.4, borderpad=0.5)
    else:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   fontsize=6, borderaxespad=0.0)


def _apply_paper_title(fig, ax, subtitle: str) -> None:
    """'Latent Program Trajectory' as a bold headline, with a smaller subtitle
    (instance / model / outcome) below it -- for the 1-2 figures picked for
    the paper, replacing the generic 'UMAP -- <instance_id> (<outcome>)' used
    while browsing candidates."""
    ax.set_title("")
    ax.annotate("Latent Program Trajectory",
                xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 24), textcoords="offset points",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                annotation_clip=False)
    ax.annotate(subtitle,
                xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color="0.25",
                annotation_clip=False)


def _strip_axes_ticks(ax) -> None:
    """The UMAP/PCA axes are in arbitrary, non-comparable units -- only the
    shape of the path matters, not the numbers -- so drop tick marks/labels."""
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{ext}")
    plt.close(fig)


def _instance_sets(trajectories: list[dict], coords_list: list[np.ndarray], overlay: bool,
                    prefix: str, color_values_list: list[np.ndarray] | None = None):
    """Yields (name, trajs, coords, color_values) groups: one group (all
    instances) if overlay, else one group per instance."""
    if overlay:
        return [(prefix, trajectories, coords_list, color_values_list)]
    cvs = color_values_list if color_values_list is not None else [None] * len(trajectories)
    return [(f"{prefix}_{t['instance_id']}", [t], [c], [cv])
            for t, c, cv in zip(trajectories, coords_list, cvs)]


def _color_values_for(trajectories: list[dict], weights: dict[str, dict],
                       color_by: str) -> list[np.ndarray] | None:
    """None for the default step-order coloring; otherwise one array per
    trajectory of that probe's exact P(positive) at each (already-subsampled)
    step, computed straight from the probe's own weights -- independent of
    whichever 2D basis a given mode happens to be plotting."""
    if color_by == "step":
        return None
    w = weights[color_by]
    return [probe_probs(t["H"], w["weight"], w["bias"], w["mean"]) for t in trajectories]


def _add_color_colorbar(fig, ax, sc, color_by: str, shrink: float = 0.8) -> None:
    if color_by == "step" or sc is None:
        return
    label = _style.PROPERTY_LABELS[PROBE_TO_PROPERTY[color_by]]
    fig.colorbar(sc, ax=ax, label=f"P({label})", shrink=shrink)


# ---------------------------------------------------------------------------
# Mode: per_probe
# ---------------------------------------------------------------------------

def _draw_per_probe_background(ax, coords_list: list[np.ndarray], norm_d: float, bias_diff: float):
    xs, ys, extent = _grid_for_coords(coords_list)
    gx, gy = np.meshgrid(xs, ys)
    # Exact logit difference depends only on the x (axis-1) coordinate.
    logit_diff = gx * norm_d + bias_diff
    prob_pos = 1.0 / (1.0 + np.exp(-logit_diff))

    cf = ax.contourf(gx, gy, prob_pos, levels=20, cmap="RdBu_r", alpha=0.25,
                      vmin=0, vmax=1, extent=extent)
    ax.contour(gx, gy, prob_pos, levels=[0.5], colors="black",
               linewidths=1.0, linestyles="--")
    return cf, extent


def plot_per_probe(trajectories: list[dict], weights: dict[str, dict], probes: list[str],
                    output_dir: Path, overlay: bool, color_by: str = "step") -> None:
    _style.apply()
    H_pooled = np.concatenate([t["H"] for t in trajectories], axis=0)
    for probe in probes:
        w = weights[probe]
        basis, norm_d, bias_diff, explained_y = per_probe_basis(w["weight"], w["bias"], w["mean"], H_pooled)
        label = _style.PROPERTY_LABELS[PROBE_TO_PROPERTY[probe]]
        coords_list = [project(t["H"], basis, w["mean"]) for t in trajectories]
        color_values_list = _color_values_for(trajectories, weights, color_by)
        cmap = "viridis" if color_by == "step" else "RdBu_r"
        # Background already gives an exact P(positive) colorbar for THIS
        # probe -- skip a redundant second one if coloring points by the same probe.
        show_point_colorbar = color_by not in ("step", probe)

        for name, trajs, cl, cv in _instance_sets(trajectories, coords_list, overlay,
                                                   f"per_probe_{probe}", color_values_list):
            fig, ax = plt.subplots(figsize=(_style.HALF_WIDTH * 1.6, _style.HALF_WIDTH))
            cf, extent = _draw_per_probe_background(ax, coords_list, norm_d, bias_diff)
            fig.colorbar(cf, ax=ax, label="P(positive)", shrink=0.8)

            if overlay:
                sc = _plot_trajectories(ax, trajs, cl, color_values_list=cv, cmap=cmap)
                title = f"{label} — exact axis + residual PC1"
            else:
                sc = _plot_single_trajectory(ax, trajs[0], cl[0], color_values=cv[0], cmap=cmap)
                outcome = "resolved" if trajs[0]["outcome"] else "unresolved"
                title = f"{label} — {_short_id(trajs[0]['instance_id'])} ({outcome})"
            if show_point_colorbar:
                _add_color_colorbar(fig, ax, sc, color_by)

            # Percentile-based limits: don't let a few extreme-norm outlier
            # tokens (common in LLM activations) stretch the axes.
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel(f"{label} direction (exact)")
            ax.set_ylabel(f"Residual PC1 ({explained_y:.0%} var, orthogonal to x)")
            ax.set_title(title)
            _finalize_legend(ax)
            fig.tight_layout()
            _save(fig, output_dir, name)


# ---------------------------------------------------------------------------
# Mode: combined
# ---------------------------------------------------------------------------

def _draw_projected_boundaries(ax, gx, gy, coords_grid, weights: dict[str, dict],
                                probes: list[str], basis: np.ndarray) -> None:
    for probe in probes:
        w = weights[probe]
        direction = w["weight"][1] - w["weight"][0]
        direction = direction / np.linalg.norm(direction)
        # Least-squares projection of this probe's direction onto the 2D basis.
        proj_direction = basis @ direction  # (2,)
        score = coords_grid @ proj_direction
        color = _style.PROPERTY_COLORS[PROBE_TO_PROPERTY[probe]]
        ax.contour(gx, gy, score.reshape(gx.shape), levels=[0.0], colors=[color],
                   linewidths=1.2, linestyles="--")
        ax.plot([], [], color=color, linestyle="--",
                label=f"{_style.PROPERTY_LABELS[PROBE_TO_PROPERTY[probe]]} (approx.)")


def plot_combined(trajectories: list[dict], weights: dict[str, dict], probes: list[str],
                   output_dir: Path, center_basis: bool, overlay: bool,
                   color_by: str = "step") -> None:
    _style.apply()
    basis, explained, W, _ = combined_basis(weights, center_basis)
    ref_mean = np.mean([weights[p]["mean"] for p in probes], axis=0)

    # Cosine similarity of each PC to each probe's own decision direction.
    print("combined mode: explained variance ratio (top-2):", explained)
    print("combined mode: PC x probe cosine similarity")
    for i, pc in enumerate(basis):
        sims = {p: float(np.dot(pc, W[j]) / (np.linalg.norm(pc) * np.linalg.norm(W[j])))
                for j, p in enumerate(probes)}
        print(f"  PC{i+1}:", {k: round(v, 3) for k, v in sims.items()})

    coords_list = [project(t["H"], basis, ref_mean) for t in trajectories]
    xs, ys, extent = _grid_for_coords(coords_list)
    gx, gy = np.meshgrid(xs, ys)
    coords_grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    color_values_list = _color_values_for(trajectories, weights, color_by)
    cmap = "viridis" if color_by == "step" else "RdBu_r"

    for name, trajs, cl, cv in _instance_sets(trajectories, coords_list, overlay, "combined",
                                               color_values_list):
        fig, ax = plt.subplots(figsize=(_style.HALF_WIDTH * 1.6, _style.HALF_WIDTH))
        _draw_projected_boundaries(ax, gx, gy, coords_grid, weights, probes, basis)

        if overlay:
            sc = _plot_trajectories(ax, trajs, cl, color_values_list=cv, cmap=cmap)
            title = "Combined probe-derived subspace"
        else:
            sc = _plot_single_trajectory(ax, trajs[0], cl[0], color_values=cv[0], cmap=cmap)
            outcome = "resolved" if trajs[0]["outcome"] else "unresolved"
            title = f"Combined probe-derived subspace — {_short_id(trajs[0]['instance_id'])} ({outcome})"
        _add_color_colorbar(fig, ax, sc, color_by)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} var)")
        ax.set_title(title)
        _finalize_legend(ax)
        fig.tight_layout()
        _save(fig, output_dir, name)


# ---------------------------------------------------------------------------
# Mode: pca
# ---------------------------------------------------------------------------

def plot_pca(trajectories: list[dict], weights: dict[str, dict], probes: list[str],
             output_dir: Path, overlay: bool, color_by: str = "step") -> None:
    from sklearn.decomposition import PCA

    _style.apply()
    all_H = np.concatenate([t["H"] for t in trajectories], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_H)
    basis = pca.components_  # (2, hidden_dim)
    mean = pca.mean_
    explained = pca.explained_variance_ratio_

    coords_list = [project(t["H"], basis, mean) for t in trajectories]
    xs, ys, extent = _grid_for_coords(coords_list)
    gx, gy = np.meshgrid(xs, ys)
    coords_grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    color_values_list = _color_values_for(trajectories, weights, color_by)
    cmap = "viridis" if color_by == "step" else "RdBu_r"

    for name, trajs, cl, cv in _instance_sets(trajectories, coords_list, overlay, "pca",
                                               color_values_list):
        fig, ax = plt.subplots(figsize=(_style.HALF_WIDTH * 1.6, _style.HALF_WIDTH))
        _draw_projected_boundaries(ax, gx, gy, coords_grid, weights, probes, basis)

        if overlay:
            sc = _plot_trajectories(ax, trajs, cl, color_values_list=cv, cmap=cmap)
            title = "Unsupervised PCA (no probe information)"
        else:
            sc = _plot_single_trajectory(ax, trajs[0], cl[0], color_values=cv[0], cmap=cmap)
            outcome = "resolved" if trajs[0]["outcome"] else "unresolved"
            title = f"Unsupervised PCA — {_short_id(trajs[0]['instance_id'])} ({outcome})"
        _add_color_colorbar(fig, ax, sc, color_by)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} var)")
        ax.set_title(title)
        _finalize_legend(ax)
        fig.tight_layout()
        _save(fig, output_dir, name)


# ---------------------------------------------------------------------------
# Mode: umap
# ---------------------------------------------------------------------------

def plot_umap(trajectories: list[dict], weights: dict[str, dict], output_dir: Path,
              overlay: bool, color_by: str = "step", paper: bool = False,
              model_label: str = "", model_key: str = "") -> None:
    import umap

    _style.apply()
    all_H = np.concatenate([t["H"] for t in trajectories], axis=0)
    n_neighbors = max(2, min(15, len(all_H) - 1))
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(all_H)

    coords_list = []
    i = 0
    for t in trajectories:
        n = len(t["H"])
        coords_list.append(embedding[i:i + n])
        i += n

    xs, ys, extent = _grid_for_coords(coords_list)
    color_values_list = _color_values_for(trajectories, weights, color_by)
    cmap = _time_cmap(model_key) if color_by == "step" else "RdBu_r"

    for name, trajs, cl, cv in _instance_sets(trajectories, coords_list, overlay, "umap",
                                               color_values_list):
        fig, ax = plt.subplots(figsize=(_style.HALF_WIDTH * 1.6, _style.HALF_WIDTH))

        if overlay:
            sc = _plot_trajectories(ax, trajs, cl, color_values_list=cv, cmap=cmap)
            title = "UMAP (no probe information)"
        else:
            sc = _plot_single_trajectory(ax, trajs[0], cl[0], color_values=cv[0], cmap=cmap)
            outcome = "resolved" if trajs[0]["outcome"] else "unresolved"
            title = f"UMAP — {_short_id(trajs[0]['instance_id'])} ({outcome})"
        _add_color_colorbar(fig, ax, sc, color_by, shrink=0.6 if paper else 0.8)
        if paper and color_by == "step":
            _add_time_swatch(ax, cmap)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if paper:
            _strip_axes_ticks(ax)
            ax.set_xlabel("UMAP dim. 1")
            ax.set_ylabel("UMAP dim. 2")
            subtitle_parts = [_short_id(trajs[0]["instance_id"])] if not overlay else []
            if model_label:
                subtitle_parts.append(model_label)
            if not overlay:
                subtitle_parts.append("resolved" if trajs[0]["outcome"] else "unresolved")
            _apply_paper_title(fig, ax, " — ".join(subtitle_parts))
        else:
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.set_title(title)
        _finalize_legend(ax, paper=paper)
        fig.tight_layout()
        if paper:
            fig.subplots_adjust(top=0.82)  # room for the two-line headline/subtitle
        _save(fig, output_dir, name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="laguna_xs2_full",
                         help="Generation/activation run id under outputs/swebench/")
    parser.add_argument("--instances", nargs="+", required=True,
                         help="instance_id[s] (filenames under outputs/swebench/<run-id>/, without .pt)")
    parser.add_argument("--probe-run-id", default="laguna_xs2_full_pooled",
                         help="Probe weights run id under results/swebench/")
    parser.add_argument("--probes", nargs="+", default=ALL_PROBES, choices=ALL_PROBES)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--mode", nargs="+", default=["per_probe", "combined", "pca", "umap"],
                         choices=["per_probe", "combined", "pca", "umap"])
    parser.add_argument("--center-basis", action="store_true",
                         help="(combined mode) subtract the mean of the 4 probe directions before SVD")
    parser.add_argument("--color-by", default="step", choices=["step"] + ALL_PROBES,
                         help="Color each point by step order (default) or by a probe's exact "
                              "P(positive) at that hidden state (computed from the probe's own "
                              "weights, independent of whichever 2D basis is being plotted).")
    parser.add_argument("--overlay", action="store_true",
                         help="Overlay all --instances on one figure per mode, instead of the "
                              "default one figure per instance (basis/background is still fit "
                              "jointly across --instances either way, for comparable axes).")
    parser.add_argument("--after-edit-only", action="store_true",
                         help="Subsample each trajectory to the first captured step after each "
                              "real code edit (plus start/end), instead of every stride-5-token "
                              "step. Requires cache/<dataset>/<run-id>/edit_step_index.pt (see "
                              "build_edit_step_index.py).")
    parser.add_argument("--activations-dir", default="outputs/swebench")
    parser.add_argument("--results-dir", default="results/swebench")
    parser.add_argument("--cache-dir", default="cache",
                         help="Base cache dir containing <dataset>/<run-id>/edit_step_index.pt")
    parser.add_argument("--output-dir", default="paper/figures/trajectories")
    parser.add_argument("--paper", action="store_true",
                         help="Paper-ready styling for the 1-2 figures picked for inclusion: no "
                              "axis ticks (units are arbitrary), 'Latent Program Trajectory' "
                              "headline + instance/model subtitle instead of the generic browsing "
                              "title, legend moved inside the axes so it doesn't collide with the "
                              "colorbar. Only affects --mode umap for now.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    activations_dir = Path(args.activations_dir)
    weights = _load_probe_weights(Path(args.results_dir), args.probe_run_id, args.layer)
    model_key = _model_key(args.run_id)
    model_label = MODEL_LABELS.get(model_key, args.run_id)

    edit_step_index = None
    if args.after_edit_only:
        edit_step_index = _load_edit_step_index(Path(args.cache_dir), activations_dir, args.run_id)

    trajectories = [
        _load_trajectory(activations_dir, args.run_id, instance_id, args.layer,
                          after_edit_only=args.after_edit_only, edit_step_index=edit_step_index)
        for instance_id in args.instances
    ]

    if "per_probe" in args.mode:
        plot_per_probe(trajectories, weights, args.probes, output_dir, args.overlay, args.color_by)
    if "combined" in args.mode:
        plot_combined(trajectories, weights, args.probes, output_dir, args.center_basis,
                       args.overlay, args.color_by)
    if "pca" in args.mode:
        plot_pca(trajectories, weights, args.probes, output_dir, args.overlay, args.color_by)
    if "umap" in args.mode:
        plot_umap(trajectories, weights, output_dir, args.overlay, args.color_by,
                   paper=args.paper, model_label=model_label, model_key=model_key)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
