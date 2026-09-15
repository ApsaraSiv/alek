#!/usr/bin/env python3
"""One-shot render of the drift_viz overlay (same topics RViz shows) to PNG."""
import math
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Path

from erc_solution import navigation_node as nav

SHELF_FRONT_WORLD_X = 2.755
SHELF_HALF_WIDTH = 2.5
BIN_WORLD = (-1.0, 0.0)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/drift_overlay.png'
    title_note = sys.argv[2] if len(sys.argv) > 2 else ''
    rclpy.init()
    node = Node('drift_render', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    got = {}
    node.create_subscription(Path, '/drift_viz/odom_path', lambda m: got.__setitem__('odom', m), 5)
    node.create_subscription(Path, '/drift_viz/true_path', lambda m: got.__setitem__('true', m), 5)
    deadline = time.time() + 8
    while time.time() < deadline and not ('odom' in got and 'true' in got):
        rclpy.spin_once(node, timeout_sec=0.1)

    def xy(path):
        return ([p.pose.position.x for p in path.poses], [p.pose.position.y for p in path.poses]) if path else ([], [])

    ox, oy = xy(got.get('odom'))
    tx, ty = xy(got.get('true'))

    bg, fg, grid = '#1f2226', '#e8e8e8', '#3a3f46'
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=110)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.tick_params(colors=fg, labelsize=9)
    ax.grid(True, color=grid, linewidth=0.6)
    ax.set_aspect('equal')

    sx = [nav.world_xy_to_odom(SHELF_FRONT_WORLD_X, wy)[0] for wy in (-SHELF_HALF_WIDTH, SHELF_HALF_WIDTH)]
    sy = [nav.world_xy_to_odom(SHELF_FRONT_WORLD_X, wy)[1] for wy in (-SHELF_HALF_WIDTH, SHELF_HALF_WIDTH)]
    ax.plot(sx, sy, color='#b8b8c4', linewidth=4, solid_capstyle='butt', label='shelf front')

    bin_ox, bin_oy = nav.world_xy_to_odom(*BIN_WORLD)
    ax.plot(bin_ox, bin_oy, marker='s', markersize=13, color='#d64545', linestyle='none', label='collection bin')

    bx, by, byaw = nav.BIN_WAYPOINT
    ax.add_patch(plt.Circle((bx, by), nav.POSITION_TOLERANCE, color='#ff6b6b', alpha=0.15, zorder=1))
    ax.add_patch(plt.Circle((bx, by), nav.POSITION_TOLERANCE, fill=False, color='#ff6b6b', linewidth=1.2, zorder=1))
    ax.arrow(bx, by, 0.35 * math.cos(byaw), 0.35 * math.sin(byaw), width=0.03, color='#ff6b6b', zorder=4)
    ax.plot([], [], color='#ff6b6b', marker='>', linestyle='none',
            label=f'planned bin waypoint (±{nav.POSITION_TOLERANCE} m)')

    ax.plot(0, 0, marker='o', color=fg, markersize=6, linestyle='none', label='spawn / odom origin')

    if ox:
        ax.plot(ox, oy, color='#ffa024', linewidth=2.2, label='odometry path (/odom)', zorder=3)
        ax.plot(ox[-1], oy[-1], marker='o', color='#ffa024', markersize=7, zorder=5)
    if tx:
        ax.plot(tx, ty, color='#34d27a', linewidth=2.2, label='ground truth path (Gazebo)', zorder=3)
        ax.plot(tx[-1], ty[-1], marker='o', color='#34d27a', markersize=7, zorder=5)

    drift_txt = ''
    if ox and tx:
        drift = math.hypot(ox[-1] - tx[-1], oy[-1] - ty[-1])
        ax.plot([ox[-1], tx[-1]], [oy[-1], ty[-1]], color='#ff3b3b', linewidth=2, zorder=6)
        drift_txt = f'current drift {drift:.2f} m'
        ax.annotate(drift_txt, ((ox[-1] + tx[-1]) / 2, (oy[-1] + ty[-1]) / 2), color='#ff6b6b',
                    fontsize=11, fontweight='bold', xytext=(10, 10), textcoords='offset points')

    xs = sx + [bin_ox, bx, 0.0] + ox + tx
    ys = sy + [bin_oy, by, 0.0] + oy + ty
    pad = 0.6
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel('odom x (m)', color=fg)
    ax.set_ylabel('odom y (m)', color=fg)
    ax.set_title('Planned waypoints vs odometry vs ground truth (odom frame)'
                 + (f'\n{title_note}' if title_note else ''), color=fg, fontsize=12)
    leg = ax.legend(loc='upper right', fontsize=8.5, facecolor='#2a2e33', edgecolor=grid)
    for t in leg.get_texts():
        t.set_color(fg)
    fig.text(0.01, 0.005, f'rendered from /drift_viz topics (same data as RViz); odom pts={len(ox)} truth pts={len(tx)}',
             color='#9aa0a6', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, facecolor=bg)
    print('wrote', out, '|', drift_txt or 'no drift yet', f'| odom pts={len(ox)} truth pts={len(tx)}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
