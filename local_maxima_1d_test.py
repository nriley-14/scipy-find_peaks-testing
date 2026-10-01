import numpy as np
from scipy.signal._peak_finding_utils import _local_maxima_1d

VIEW_GRAPHS = True  # Set False to skip writing the graphs.
OUTPUT_HTML = "local_maxima_tests.html"
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"

cases = [
    # name, input, wrap, expected (midpoints, left_edges, right_edges)
    ("single peak", [0, 1, 0], False, ([1], [1], [1])),
    ("single plateau", [0, 1, 1, 1, 0], False, ([2], [1], [3])),
    ("cos wrap", np.cos(np.linspace(0, 3 * np.pi, 50)),
     True, ([0, 33], [0, 33], [0, 33])),
    ("cos no wrap", np.cos(np.linspace(0, 3 * np.pi, 50)),
     False, ([33], [33], [33])),
    ("flat line", [0, 1, 1, 0], True, ([1], [1], [2])),
    ("right peak wrap", [1, 0, 1, 2], True, ([3], [3], [3])),
    ("right peak no wrap", [1, 0, 1, 2], False, ([], [], [])),
    ("right and left equal wrap", [1, 0, 1], True, ([2], [2], [0])),
    ("right and left equal no wrap", [1, 0, 1], False, ([], [], [])),
    ("w-wrap", [1, 1, 0, 0, 1, 1], True, ([5], [4], [1])),
    ("last sample peak", [0, 0, 1], True, ([2], [2], [2])),
    ("no_wrap_plateu", [0, 1, 1], False, ([], [], [])),
    ("wrap_plateu", [0, 1, 1], True, ([1], [1], [2])),
    ("two samples wrap", [0, 1], True, ([1], [1], [1])),
    ("plateau at end", [0, 0, 1, 1], True, ([2], [2], [3])),
    ("alternating peaks", [0, 1, 0, 1], True, ([1, 3], [1, 3], [1, 3])),
    ("peak then end plateau", [0, 1, 0, 0, 2, 2],
     True, ([1, 4], [1, 4], [1, 5])),
    ("interior and boundary peaks", [1, 0, 2, 0, 1],
     True, ([2, 4], [2, 4], [2, 0])),
]


def run_cases(cases):
    """Run each case and report its actual peak indices."""
    results = []
    num_pass = 0

    for name, values, wrap, expected in cases:
        y = np.asarray(values, dtype=np.float64)
        actual = tuple(a.tolist() for a in _local_maxima_1d(y, wrap=wrap))
        results.append((name, y, wrap, actual))

        passed = actual == expected
        num_pass += passed
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
        if not passed:
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")

    print(f"Num pass: {num_pass}, Num fail: {len(results) - num_pass}")
    return results


def marker_offsets(groups, size):
    """Separate overlapping roles visually while keeping their true indices."""
    offsets = [{} for _ in groups]
    for index in sorted({i for group in groups for i in group}):
        roles = [role for role, group in enumerate(groups) if index in group]
        if len(roles) > 1:
            extent = 0.015 * (size - 1)
            for role, offset in zip(roles, np.linspace(-extent, extent, len(roles))):
                offsets[role][index] = offset
    return offsets


def plot_results(results, output_path=OUTPUT_HTML):
    """Write the signals, wrapped neighbors, and detected peaks to HTML."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=len(results),
        cols=1,
        subplot_titles=[name for name, *_ in results],
        vertical_spacing=0.04,
    )
    legend_seen = set()

    def add_trace(row, name, **kwargs):
        fig.add_trace(
            go.Scatter(
                name=name,
                legendgroup=name,
                showlegend=name not in legend_seen,
                **kwargs,
            ),
            row=row,
            col=1,
        )
        legend_seen.add(name)

    for row, (name, y, wrap, actual) in enumerate(results, start=1):
        midpoints, left_edges, right_edges = actual
        size = len(y)
        groups = (left_edges, midpoints, right_edges)
        offsets = marker_offsets(groups, size)

        add_trace(
            row, "Array",
            x=np.arange(size), y=y,
            mode="lines+markers",
            line=dict(color=BLUE),
            marker=dict(color=BLUE),
        )

        if wrap and size:
            add_trace(
                row, "Wraparound",
                x=[-1, 0, None, size - 1, size],
                y=[y[-1], y[0], None, y[-1], y[0]],
                mode="lines",
                line=dict(color=ORANGE),
            )

        if midpoints:
            # Guide lines remain at the true sample positions.
            for index in sorted(set(left_edges + midpoints + right_edges)):
                fig.add_vline(
                    x=index,
                    line_color=BLUE,
                    line_dash="dot",
                    line_width=1,
                    opacity=0.35,
                    row=row,
                    col=1,
                )

            roles = (
                ("Peak left edge", "triangle-left"),
                ("Peak midpoint", "diamond"),
                ("Peak right edge", "triangle-right"),
            )
            for indices, role_offsets, (label, symbol) in zip(groups, offsets, roles):
                add_trace(
                    row, label,
                    x=[i + role_offsets.get(i, 0) for i in indices],
                    y=y[indices],
                    mode="markers",
                    text=[f"Index {i}" for i in indices],
                    hovertemplate=f"%{{text}}<extra>{label}</extra>",
                    marker=dict(
                        color=BLUE,
                        symbol=symbol,
                        size=16,
                        line=dict(color="white", width=1),
                    ),
                )

        fig.update_xaxes(row=row, col=1, title_text="Index")
        fig.update_yaxes(row=row, col=1, title_text="Value")

    fig.update_layout(
        height=300 * len(results),
        title_text="Local maxima test cases",
        template="plotly_white",
    )
    fig.write_html(output_path, auto_open=False)
    print(f"Wrote {output_path}")


def main():
    results = run_cases(cases)
    if VIEW_GRAPHS:
        plot_results(results)


if __name__ == "__main__":
    main()

