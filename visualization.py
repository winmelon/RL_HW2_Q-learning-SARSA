"""
Visualization module: generates all plots for the experiment report.
Saves figures to the 'figures/' directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4d',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#e0e0e0',
    'ytick.color':      '#e0e0e0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'legend.framealpha': 0.2,
    'legend.edgecolor': '#555',
})

Q_COLOR   = '#00d4ff'   # cyan-blue  for Q-learning
S_COLOR   = '#ff6b6b'   # coral-red  for SARSA
CLIFF_CLR = '#ff4444'
GOAL_CLR  = '#00ff88'
START_CLR = '#ffaa00'

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

ACTION_ARROWS = {0: (0, 0.35), 1: (0, -0.35), 2: (-0.35, 0), 3: (0.35, 0)}
ACTION_NAMES  = ['↑', '↓', '←', '→']


# ── helper ─────────────────────────────────────────────────────────────────────

def smooth(data, window=20):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {path}')
    return path


# ── 1. Reward curves ───────────────────────────────────────────────────────────

def plot_reward_curves(ql_rewards, sarsa_rewards, window=20):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Episode Reward Curves: Q-Learning vs SARSA', fontsize=15, fontweight='bold', y=1.01)

    n_ep = len(ql_rewards)
    x_raw = np.arange(n_ep)
    x_sm  = np.arange(window - 1, n_ep)

    # ── raw (transparent) + smoothed ──
    for ax, rewards, color, label in [
        (axes[0], ql_rewards, Q_COLOR, 'Q-Learning'),
        (axes[1], sarsa_rewards, S_COLOR, 'SARSA'),
    ]:
        ax.plot(x_raw, rewards, alpha=0.20, color=color, linewidth=0.8)
        ax.plot(x_sm, smooth(rewards, window), color=color, linewidth=2.2,
                label=f'{label} (smoothed, w={window})')
        ax.axhline(np.mean(rewards[-100:]), color=color, linestyle=':', linewidth=1.5,
                   label=f'Last-100 mean: {np.mean(rewards[-100:]):.1f}')
        ax.set_ylabel('Total Reward', labelpad=8)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True)
        ax.set_title(label, color=color, fontweight='bold')
        ax.set_ylim(bottom=max(min(rewards) - 20, -600))

    axes[1].set_xlabel('Episode')
    plt.tight_layout()
    return _save(fig, '1_reward_curves.png')


# ── 2. Reward comparison (overlay) ─────────────────────────────────────────────

def plot_reward_comparison(ql_rewards, sarsa_rewards, window=20):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_ep = len(ql_rewards)
    x_sm = np.arange(window - 1, n_ep)

    ax.plot(x_sm, smooth(ql_rewards, window),   color=Q_COLOR, linewidth=2.2, label='Q-Learning')
    ax.plot(x_sm, smooth(sarsa_rewards, window), color=S_COLOR, linewidth=2.2, label='SARSA')

    ax.fill_between(x_sm, smooth(ql_rewards, window), smooth(sarsa_rewards, window),
                    alpha=0.12, color='white')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (smoothed)')
    ax.set_title('Q-Learning vs SARSA — Smoothed Reward Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return _save(fig, '2_reward_comparison.png')


# ── 3. Rolling std (stability) ─────────────────────────────────────────────────

def plot_stability(ql_rewards, sarsa_rewards, window=30):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_ep = len(ql_rewards)

    def rolling_std(arr, w):
        return [np.std(arr[max(0, i - w):i + 1]) for i in range(n_ep)]

    ql_std = rolling_std(ql_rewards, window)
    sa_std = rolling_std(sarsa_rewards, window)

    ax.plot(ql_std, color=Q_COLOR, linewidth=2, label='Q-Learning σ')
    ax.plot(sa_std, color=S_COLOR, linewidth=2, label='SARSA σ')
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Rolling Std (window={window})')
    ax.set_title('Learning Stability: Rolling Standard Deviation of Rewards', fontweight='bold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return _save(fig, '3_stability.png')


# ── 4. Policy grid ─────────────────────────────────────────────────────────────

def _draw_policy_grid(ax, Q_table, env, path, title, color):
    H, W = env.height, env.width

    # Background cell colors
    cell_colors = np.full((H, W, 3), [26/255, 29/255, 39/255])
    for r in range(H):
        for c in range(W):
            pos = (r, c)
            if pos in env.cliff:
                cell_colors[r, c] = [0.8, 0.1, 0.1]
            elif pos == env.goal:
                cell_colors[r, c] = [0.0, 0.7, 0.3]
            elif pos == env.start:
                cell_colors[r, c] = [0.9, 0.55, 0.0]

    ax.imshow(cell_colors, aspect='auto', interpolation='nearest')

    # Draw policy arrows
    policy = np.argmax(Q_table, axis=1)
    for idx in range(env.n_states):
        r = idx // W
        c = idx % W
        pos = (r, c)
        if pos in env.cliff:
            ax.text(c, r, 'CLIFF', ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold')
            continue
        if pos == env.goal:
            ax.text(c, r, 'GOAL', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')
            continue
        if pos == env.start:
            ax.text(c, r, 'START', ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold')
            continue

        a = policy[idx]
        dx, dy = ACTION_ARROWS[a]
        ax.annotate('', xy=(c + dx, r + dy), xytext=(c, r),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Draw learned path
    if path and len(path) > 1:
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, color='yellow', linewidth=2.5, alpha=0.85,
                marker='o', markersize=4, zorder=5, label='Greedy path')

    # Grid lines
    for x in range(W + 1):
        ax.axvline(x - 0.5, color='#3a3d4d', linewidth=0.8)
    for y in range(H + 1):
        ax.axhline(y - 0.5, color='#3a3d4d', linewidth=0.8)

    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xticklabels(range(W), fontsize=8)
    ax.set_yticklabels(range(H), fontsize=8)
    ax.set_title(title, color=color, fontsize=13, fontweight='bold', pad=8)


def plot_policy_grids(ql_agent, sarsa_agent, env, ql_path, sarsa_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle('Learned Policy & Greedy Path', fontsize=15, fontweight='bold')

    _draw_policy_grid(axes[0], ql_agent.Q, env, ql_path,    'Q-Learning Policy', Q_COLOR)
    _draw_policy_grid(axes[1], sarsa_agent.Q, env, sarsa_path, 'SARSA Policy',    S_COLOR)

    legend_elements = [
        mpatches.Patch(color=[0.8, 0.1, 0.1], label='Cliff'),
        mpatches.Patch(color=[0.0, 0.7, 0.3], label='Goal'),
        mpatches.Patch(color=[0.9, 0.55, 0.0], label='Start'),
        plt.Line2D([0], [0], color='yellow', linewidth=2, marker='o', label='Greedy Path'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.08), fontsize=10)

    plt.tight_layout()
    return _save(fig, '4_policy_grid.png')


# ── 5. Q-value heat maps ───────────────────────────────────────────────────────

def plot_qvalue_heatmaps(ql_agent, sarsa_agent, env):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle('Q-Value Heatmaps (max over actions)', fontsize=14, fontweight='bold')
    H, W = env.height, env.width

    cmap_ql = LinearSegmentedColormap.from_list('ql', ['#0a0a1a', Q_COLOR])
    cmap_sa = LinearSegmentedColormap.from_list('sa', ['#0a0a1a', S_COLOR])

    def _make_grid(Q):
        grid = np.max(Q, axis=1).reshape(H, W)
        # Mask cliff
        for r in range(H):
            for c in range(W):
                if (r, c) in env.cliff:
                    grid[r, c] = np.nan
        return grid

    agents = [
        (ql_agent, cmap_ql, Q_COLOR, 'Q-Learning'),
        (sarsa_agent, cmap_sa, S_COLOR, 'SARSA'),
    ]

    for col, (agent, cmap, color, name) in enumerate(agents):
        grid = _make_grid(agent.Q)
        im = axes[0, col].imshow(grid, cmap=cmap, aspect='auto', interpolation='nearest')
        axes[0, col].set_title(f'{name} — max Q(s,a)', color=color, fontweight='bold')
        plt.colorbar(im, ax=axes[0, col])
        _annotate_special_cells(axes[0, col], env)

        # Action with highest Q value for each cell
        policy_grid = np.argmax(agent.Q, axis=1).reshape(H, W).astype(float)
        for r in range(H):
            for c in range(W):
                if (r, c) in env.cliff:
                    policy_grid[r, c] = np.nan
        cmap2 = plt.cm.get_cmap('tab10', 4)
        im2 = axes[1, col].imshow(policy_grid, cmap=cmap2, aspect='auto',
                                   interpolation='nearest', vmin=0, vmax=3)
        axes[1, col].set_title(f'{name} — Best Action per Cell', color=color, fontweight='bold')
        cbar = plt.colorbar(im2, ax=axes[1, col], ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['↑ Up', '↓ Down', '← Left', '→ Right'])
        _annotate_special_cells(axes[1, col], env)

    plt.tight_layout()
    return _save(fig, '5_qvalue_heatmaps.png')


def _annotate_special_cells(ax, env):
    H, W = env.height, env.width
    for r in range(H):
        for c in range(W):
            pos = (r, c)
            if pos in env.cliff:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color=[0.7, 0.1, 0.1], alpha=0.8))
                ax.text(c, r, 'C', ha='center', va='center', color='white', fontsize=7)
            elif pos == env.goal:
                ax.text(c, r, 'G', ha='center', va='center', color='lime', fontsize=9, fontweight='bold')
            elif pos == env.start:
                ax.text(c, r, 'S', ha='center', va='center', color='orange', fontsize=9, fontweight='bold')


# ── 6. Episode length ─────────────────────────────────────────────────────────

def plot_episode_lengths(ql_lengths, sarsa_lengths, window=20):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_ep = len(ql_lengths)
    x_sm = np.arange(window - 1, n_ep)

    ax.plot(x_sm, smooth(ql_lengths, window),   color=Q_COLOR, linewidth=2.2, label='Q-Learning')
    ax.plot(x_sm, smooth(sarsa_lengths, window), color=S_COLOR, linewidth=2.2, label='SARSA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per Episode (smoothed)')
    ax.set_title('Episode Length: Q-Learning vs SARSA', fontweight='bold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return _save(fig, '6_episode_lengths.png')


# ── 7. Summary bar chart ───────────────────────────────────────────────────────

def plot_summary_stats(ql_rewards, sarsa_rewards):
    last = 100
    ql_last   = ql_rewards[-last:]
    sarsa_last = sarsa_rewards[-last:]

    metrics = {
        'Mean Reward\n(last 100)': (np.mean(ql_last), np.mean(sarsa_last)),
        'Max Reward\n(last 100)': (np.max(ql_last), np.max(sarsa_last)),
        'Std Dev\n(last 100)': (np.std(ql_last), np.std(sarsa_last)),
    }

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, [v[0] for v in metrics.values()],
                   width, label='Q-Learning', color=Q_COLOR, alpha=0.85)
    bars2 = ax.bar(x + width/2, [v[1] for v in metrics.values()],
                   width, label='SARSA', color=S_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.set_title('Summary Statistics (Last 100 Episodes)', fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y')

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, color=Q_COLOR)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, color=S_COLOR)

    plt.tight_layout()
    return _save(fig, '7_summary_stats.png')
