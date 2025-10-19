import seaborn as sns

def format_plot(ax):
    """
    """
    sns.set_style("whitegrid")
    
    font_title = {
        'size': 16, 
        'weight': 'bold', 
        'name': 'monospace'
    }

    font_axes = {
        'size': 14, 
        'weight': 'bold', 
        'name': 'monospace'
    }

    ax.grid(True, linestyle=":", alpha=0.6)
    sns.despine(ax=ax)

    if ax.get_legend():
        ax.legend(bbox_to_anchor=(1.1, 1))
    
    ax.set_title(f"\n\n{ax.get_title()}\n", fontdict=font_title)
    ax.set_xlabel(f"\n{ax.get_xlabel().replace('_', ' ').upper()} ➞", fontdict=font_axes)
    ax.set_ylabel(f"{ax.get_ylabel().replace('_', ' ').upper()} ➞\n", fontdict=font_axes)