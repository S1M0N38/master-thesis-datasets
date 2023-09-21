import argparse
from pathlib import Path

import numpy as np
import plotly.express as px
from dash import Dash, Input, Output, callback, dcc, html

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"

app = Dash(__name__)


def init_app():
    style_div = {
        "width": "100%",
        "height": "95vh",
    }
    style_plot = {
        "width": "100%",
        "height": "90vh",
        "overflow": "hidden",
    }

    # Define the layout of the app
    app.layout = html.Div(
        [
            dcc.Dropdown(options=projections, value=projections[0], id="projection"),
            dcc.Graph(id="plot", style=style_plot),
            dcc.Slider(1, len(HIERARCHY) - 1, 1, value=1, id="level"),
        ],
        style=style_div,
    )


@callback(
    Output("plot", "figure"),
    Input("projection", "value"),
    Input("level", "value"),
    prevenet_initial_call=True,
)
def update_plot(projections, level):
    x, y = np.load(PATH_ROOT / projections).T
    labels = HIERARCHY[level]
    text = [
        f"[{labels[i]}/{max(labels)}] [{i}/{len(labels) - 1}] {cls}"
        for i, cls in enumerate(classes)
    ]
    # marker = [str(label).zfill(2)[0] for label in labels]

    fig = px.scatter(
        x=x,
        y=y,
        # symbol=marker, # don't know why shows only three markers
        color=[str(label) for label in labels],
        hover_name=text,
        hover_data=[],
    )
    fig.update_traces(marker=dict(size=7))

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)

    fig.update_xaxes(
        showticklabels=False,
        showline=False,
        zeroline=False,
        range=[x_min - x_margin, x_max + x_margin],
    )

    fig.update_yaxes(
        showticklabels=False,
        showline=False,
        zeroline=False,
        range=[y_min - y_margin, y_max + y_margin],
    )

    fig.update_layout(margin=dict(l=0, r=0, t=40, b=10))

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project Encodings to 2D using UMAP",
    )

    parser.add_argument(
        "--dataset",
        choices=[d.stem for d in PATH_DATASETS.iterdir() if d.is_dir()],
        required=True,
        help="Dataset to explore",
    )

    args = parser.parse_args()

    PATH_DATASET = PATH_ROOT / "datasets" / args.dataset
    PATH_HIERARCHY = PATH_DATASET / "hierarchy" / "hierarchy.npy"
    PATH_PROJECTIONS = PATH_DATASET / "projections" / "umap"

    HIERARCHY = np.load(PATH_HIERARCHY)
    COLOR_MAP = px.colors.qualitative.Light24 * 100

    projections = [
        str(path.relative_to(PATH_ROOT))
        for path in sorted(PATH_PROJECTIONS.rglob("*.npy"))
    ]

    with open(PATH_DATASET / "classes" / "classes.txt") as f:
        classes = [cls.strip() for cls in f.readlines()]

    init_app()
    app.run_server(debug=True)
