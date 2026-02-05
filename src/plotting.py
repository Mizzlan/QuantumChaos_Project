import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def set_publication_style():
    """
    Set Matplotlib parameters to match international
    physics journal standards (PRB / PRE style).

    This function should be called before any plot.
    """

    mpl.rcParams.update({

        # Font settings (LaTeX-like serif fonts)
        "font.family": "serif",
        "mathtext.fontset": "cm",

        # Axis labels and tick sizes
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Line and marker aesthetics
        "lines.linewidth": 1.5,
        "lines.markersize": 5,

        # Tick appearance (PRB standard)
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,

        # Figure quality
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Disable grid by default (journals dislike grids)
        "axes.grid": False,
    })


def plot_chaos_transition(
    x_data,
    y_mean,
    y_err=None,
    x_label="Control Parameter",
    title="Chaos Transition",
    filename="chaos_transition.pdf"
):
    """
    Plot ⟨r⟩ versus a control parameter (W, J2, etc.)

    Parameters
    ----------
    x_data : array
        Control parameter (e.g., disorder strength W)
    y_mean : array
        Mean ⟨r⟩ values
    y_err : array or None
        Standard deviation (error bars)
    """

    set_publication_style()

    # Create figure with PRB-standard size
    plt.figure(figsize=(6, 4))

    # Plot data (with or without error bars)
    if y_err is None:
        plt.plot(
            x_data,
            y_mean,
            "ko-",
            markerfacecolor="white",
            label="Spin chain"
        )
    else:
        plt.errorbar(
            x_data,
            y_mean,
            yerr=y_err,
            fmt="ko",
            markerfacecolor="white",
            capsize=3,
            label="Spin chain"
        )

    # Universal RMT reference lines
    plt.axhline(
        0.536,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="GOE (chaotic)"
    )

    plt.axhline(
        0.386,
        color="blue",
        linestyle="--",
        linewidth=1.2,
        label="Poisson (integrable)"
    )

    # Axis labels
    plt.xlabel(x_label)
    plt.ylabel(r"Mean $\langle r \rangle$")

    # Legend (no frame, journal style)
    plt.legend(frameon=False)

    # Save as vector PDF (journals prefer this)
    plt.savefig(filename)

    # Show for interactive use
    plt.show()
