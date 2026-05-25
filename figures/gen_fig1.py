import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

BLUE_BG = '#E8F0FE'
BLUE_EDGE = '#1A73E8'
GREEN_BG = '#E6F4EA'
GREEN_EDGE = '#34A853'
ORANGE_BG = '#FEF7E0'
ORANGE_EDGE = '#F9AB00'
GRAY_BLOCK = '#F1F3F4'
GRAY_EDGE = '#9AA0A6'
RED_DASH = '#EA4335'
TEXT_DARK = '#202124'
WHITE = '#FFFFFF'

def draw_box(ax, x, y, w, h, color_bg, color_edge, label='', fontsize=9, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                          facecolor=color_bg, edgecolor=color_edge, linewidth=1.5)
    ax.add_patch(rect)
    if label:
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=TEXT_DARK)

def draw_arrow(ax, x1, y1, x2, y2, color=TEXT_DARK, lw=1.2, style='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               linestyle=style, connectionstyle='arc3,rad=0'))

def draw_dashed_arrow(ax, x1, y1, x2, y2, color=RED_DASH, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               linestyle='dashed', connectionstyle='arc3,rad=-0.2'))

# Region boxes
rect_blue = FancyBboxPatch((3.2, 4.8), 9.6, 2.8, boxstyle='round,pad=0.15',
                           facecolor=BLUE_BG, edgecolor=BLUE_EDGE, linewidth=1.0, alpha=0.5)
ax.add_patch(rect_blue)
ax.text(8.0, 7.4, 'Stage 1: MASS Signal Generation', ha='center', fontsize=10,
        fontweight='bold', color=BLUE_EDGE)

rect_green = FancyBboxPatch((3.2, 1.5), 9.6, 2.8, boxstyle='round,pad=0.15',
                            facecolor=GREEN_BG, edgecolor=GREEN_EDGE, linewidth=1.0, alpha=0.5)
ax.add_patch(rect_green)
ax.text(8.0, 4.1, 'Stage 2: Adapter-Enhanced Forward Pass', ha='center', fontsize=10,
        fontweight='bold', color=GREEN_EDGE)

rect_orange = FancyBboxPatch((13.2, 1.5), 2.4, 6.1, boxstyle='round,pad=0.15',
                             facecolor=ORANGE_BG, edgecolor=ORANGE_EDGE, linewidth=1.0, alpha=0.5)
ax.add_patch(rect_orange)
ax.text(14.4, 7.4, 'Stage 3:\nTest-Time\nOptimization', ha='center', fontsize=9,
        fontweight='bold', color=ORANGE_EDGE)

# Input image
draw_box(ax, 0.3, 3.5, 2.0, 2.0, GRAY_BLOCK, GRAY_EDGE, 'Input\nMoire Image\nI (HxWx3)', bold=True)
for i in np.linspace(0.5, 2.5, 8):
    ax.plot([0.5, 2.1], [4.2 + i*0.15, 4.2 + i*0.15], color='#D93025', lw=0.5, alpha=0.6)
    ax.plot([0.5, 2.1], [4.3 + i*0.15, 4.3 + i*0.15], color='#1A73E8', lw=0.5, alpha=0.4)

# DWT
draw_box(ax, 3.8, 5.2, 2.4, 1.8, WHITE, BLUE_EDGE, '')
ax.text(5.0, 6.75, 'DWT Haar', ha='center', fontsize=8, fontweight='bold', color=TEXT_DARK)
for si, (label, sx, sy) in enumerate([('LL', 4.0, 5.4), ('LH', 5.2, 5.4), ('HL', 4.0, 6.05), ('HH', 5.2, 6.05)]):
    ax.add_patch(FancyBboxPatch((sx, sy), 0.85, 0.45, boxstyle='round,pad=0.03',
                                facecolor='#D2E3FC', edgecolor=BLUE_EDGE, linewidth=0.5))
    ax.text(sx+0.425, sy+0.225, label, ha='center', va='center', fontsize=7, color=TEXT_DARK)

# MFD
draw_box(ax, 7.0, 5.2, 2.2, 1.8, WHITE, BLUE_EDGE, '')
ax.text(8.1, 6.75, 'MFD', ha='center', fontsize=9, fontweight='bold', color=TEXT_DARK)
ax.text(8.1, 6.25, '4-layer CNN', ha='center', fontsize=7, color='#5F6368')
ax.text(8.1, 5.95, '12->32->64->64->3', ha='center', fontsize=7, color='#5F6368')
ax.text(8.1, 5.65, 'M in [0,1]', ha='center', fontsize=7, color='#5F6368')

# Attenuation
draw_box(ax, 10.0, 5.2, 2.0, 1.8, WHITE, BLUE_EDGE, '')
ax.text(11.0, 6.75, 'Selective', ha='center', fontsize=8, fontweight='bold', color=TEXT_DARK)
ax.text(11.0, 6.40, 'Attenuation', ha='center', fontsize=8, fontweight='bold', color=TEXT_DARK)
ax.text(11.0, 5.85, "LH' = LH(1-aM)", ha='center', fontsize=7, color=TEXT_DARK)
ax.text(11.0, 5.55, "HL' = HL(1-aM)", ha='center', fontsize=7, color=TEXT_DARK)
ax.text(11.0, 5.35, 'a = 0.5', ha='center', fontsize=7, color='#5F6368')

# Pseudo-clean target
draw_box(ax, 12.5, 5.2, 1.8, 1.8, '#E8F5E9', GREEN_EDGE, 'Pseudo-Clean\nTarget I_tilde\n[-1, 1]', fontsize=8, bold=True)

# Input adapters
draw_box(ax, 3.8, 2.2, 1.2, 1.3, WHITE, GREEN_EDGE, 'FDA\nSGA', fontsize=7)
ax.text(4.4, 1.9, 'Input', ha='center', fontsize=7, color='#5F6368')

# Frozen Backbone
draw_box(ax, 5.6, 2.2, 4.0, 1.3, GRAY_BLOCK, GRAY_EDGE, 'Frozen Pre-trained Backbone f_theta', fontsize=9, bold=True)
ax.text(7.6, 2.55, '(WDNet / DDA MBCNN)', fontsize=7, color='#5F6368', ha='center')

# Output adapters
draw_box(ax, 10.2, 2.2, 1.2, 1.3, WHITE, GREEN_EDGE, 'FDA\nSGA', fontsize=7)
ax.text(10.8, 1.9, 'Output', ha='center', fontsize=7, color='#5F6368')

# Output image
draw_box(ax, 12.5, 2.2, 1.8, 2.0, WHITE, GREEN_EDGE, 'Output\nDemoi(red)\nI_hat', fontsize=8, bold=True)
for i in range(5):
    ax.plot([12.7, 14.1], [3.2 + i*0.25, 3.2 + i*0.25], color='#34A853', lw=0.5, alpha=0.4)

# Arrows - main flow
draw_arrow(ax, 2.3, 4.5, 3.7, 6.1, TEXT_DARK)
draw_arrow(ax, 6.2, 6.1, 6.9, 6.1, TEXT_DARK)
draw_arrow(ax, 9.2, 6.1, 9.9, 6.1, TEXT_DARK)
draw_arrow(ax, 12.0, 6.1, 12.4, 6.1, TEXT_DARK)
draw_arrow(ax, 2.3, 4.0, 3.7, 2.85, TEXT_DARK)
draw_arrow(ax, 5.0, 2.85, 5.5, 2.85, TEXT_DARK)
draw_arrow(ax, 9.6, 2.85, 10.1, 2.85, TEXT_DARK)
draw_arrow(ax, 11.4, 2.85, 12.4, 3.3, TEXT_DARK)

# Red dashed gradient arrows
ax.annotate('', xy=(13.65, 4.5), xytext=(14.3, 6.1),
            arrowprops=dict(arrowstyle='->', color=RED_DASH, lw=1.5, linestyle='dashed'))
ax.annotate('', xy=(13.65, 4.5), xytext=(14.3, 3.3),
            arrowprops=dict(arrowstyle='->', color=RED_DASH, lw=1.5, linestyle='dashed'))
ax.text(13.55, 4.5, 'L_MASS =\n||I_hat - I_tilde||1\n+ lambda_reg * L_reg',
        fontsize=7, color=RED_DASH, va='center', ha='right')

ax.annotate('', xy=(13.8, 2.85), xytext=(14.3, 4.1),
            arrowprops=dict(arrowstyle='->', color=RED_DASH, lw=1.5, linestyle='dashed'))
ax.text(14.9, 3.4, 'Gradient:\nUpdate phi\n(Adapters\n only!)',
        fontsize=7, color=RED_DASH, va='center')

# Footer
ax.text(0.5, 0.6, 'Gradient updates adapter parameters only  |  Backbone frozen  |  MFD frozen',
        fontsize=8, color='#5F6368')
ax.plot([0.35, 1.25], [0.55, 0.55], color=RED_DASH, lw=1.5, linestyle='dashed')

# Adapter detail
ax.text(0.5, 7.7, 'FDA: FFT_channel -> A@B^T modulation -> IFFT + residual', fontsize=6.5, color='#5F6368')
ax.text(0.5, 7.45, 'SGA: 1x1 Conv -> 3x3 DWConv -> 1x1 Conv -> sigmoid gate + residual', fontsize=6.5, color='#5F6368')

plt.tight_layout(pad=0.5)
plt.savefig('fig1_framework.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('fig1_framework.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved: fig1_framework.pdf + fig1_framework.png')
