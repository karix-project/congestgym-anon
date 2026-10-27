"""
congestgym/utils/visualization.py

Renders the network: link colour/width encodes utilization (green healthy
-> red congested), node colour/size encodes buffer occupancy. Pure
rendering -- reads NetworkModel state, mutates nothing.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import networkx as nx
import numpy as np


class NetworkVisualizer:
    def __init__(self, network_model, layout_seed: int = 0):
        self.model = network_model
        self._pos = nx.spring_layout(self.model.graph, seed=layout_seed)

    def render(self, step: int | None = None, save_path: str | None = None,
               show: bool = False, return_rgb: bool = False):
        """Draws the current network state. Returns an (H, W, 3) uint8 array
        when `return_rgb` is set, otherwise `save_path` (None if not saving)."""
        g = self.model.graph

        edge_colors, edge_widths = [], []
        for u, v in g.edges():
            util = float(np.clip(g[u][v].get("utilization", 0.0), 0.0, 1.0))
            edge_colors.append((util, 1.0 - util, 0.15))
            edge_widths.append(1.0 + 4.0 * util)

        occ = self.model.get_node_occupancy()
        node_util = np.clip(occ / self.model.get_node_buffers(), 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(7.5, 6))
        nx.draw_networkx_edges(g, self._pos, edge_color=edge_colors,
                               width=edge_widths, ax=ax)
        nodes = nx.draw_networkx_nodes(
            g, self._pos, node_color=node_util, cmap="RdYlGn_r",
            node_size=350 + 750 * node_util, vmin=0, vmax=1,
            edgecolors="black", linewidths=0.6, ax=ax)
        nx.draw_networkx_labels(g, self._pos, font_size=8, ax=ax)

        fig.colorbar(nodes, ax=ax, label="node buffer occupancy", shrink=0.8)
        title = "CongestGym network state"
        if step is not None:
            title += f"  (step {step})"
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        frame = None
        if return_rgb:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        elif show:
            plt.show()
        plt.close(fig)
        return frame if return_rgb else save_path
