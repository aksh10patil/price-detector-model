import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None
from pricer.evaluation.evaluator import EvaluationResult
from pricer.evaluation.metrics import calculate_confidence_interval


def plot_scatter(result: EvaluationResult):
    """Generates an interactive scatter plot of predicted vs actual prices."""
    colors = []
    for g, t in zip(result.guesses, result.truths):
        err = abs(g - t)
        if err < 40 or err / max(t, 1.0) < 0.2:
            colors.append("green")
        elif err < 80 or err / max(t, 1.0) < 0.4:
            colors.append("orange")
        else:
            colors.append("red")

    df = pd.DataFrame(
        {
            "truth": result.truths,
            "guess": result.guesses,
            "title": result.titles,
            "color": colors,
        }
    )
    df["hover"] = [
        f"{t}<br>Predicted: ${g:,.2f}<br>Actual: ${y:,.2f}"
        for t, g, y in zip(df["title"], df["guess"], df["truth"])
    ]

    max_val = float(max(max(result.truths), max(result.guesses), 100))

    fig = px.scatter(
        df,
        x="truth",
        y="guess",
        color="color",
        color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
        title=f"{result.model_name}: Predicted vs Actual (RMSE: ${result.metrics['rmse']:,.2f}, R²: {result.metrics['r2']:.1f}%)",
        labels={"truth": "Actual Price ($)", "guess": "Predicted Price ($)"},
        width=800,
        height=600,
    )

    for tr in fig.data:
        mask = df["color"] == tr.name
        tr.customdata = df.loc[mask, ["hover"]].to_numpy()
        tr.hovertemplate = "%{customdata[0]}<extra></extra>"
        tr.marker.update(size=6)

    # Reference y = x line
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(width=2, dash="dash", color="deepskyblue"),
            name="Ideal (y = x)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[0, max_val])
    fig.update_layout(showlegend=False, template="plotly_white")
    return fig


def plot_error_trend(result: EvaluationResult):
    """Plots running cumulative average error with 95% confidence interval shaded band."""
    running_means, ci, final_mean, final_ci = calculate_confidence_interval(result.errors)
    x = list(range(1, len(result.errors) + 1))
    upper = [m + c for m, c in zip(running_means, ci)]
    lower = [m - c for m, c in zip(running_means, ci)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(128,128,128,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="95% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=running_means,
            mode="lines",
            line=dict(width=3, color="firebrick"),
            name="Cumulative Avg Error",
            customdata=list(zip(ci)),
            hovertemplate="Sample: %{x}<br>Avg Error: $%{y:,.2f}<br>±95% CI: $%{customdata[0]:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{result.model_name} Error Trend: ${final_mean:,.2f} ± ${final_ci:,.2f}",
        xaxis_title="Number of Items Evaluated",
        yaxis_title="Error ($)",
        width=800,
        height=320,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def plot_leaderboard(results: list[tuple[str, str, float]]):
    """Generates the official capstone leaderboard bar chart."""
    labels, colors, values = zip(*results)
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(
        title="The Price Is Right: Model Benchmark Leaderboard (RMSE in USD)",
        yaxis=dict(range=[0, max(values) + 10], title="RMSE ($) - Lower is Better"),
        xaxis=dict(tickangle=-45),
        width=1000,
        height=600,
        template="plotly_white",
    )
    return fig
