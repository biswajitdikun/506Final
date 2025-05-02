
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Load your cleaned dataset
df = pd.read_csv("data.csv")
df['open_dt'] = pd.to_datetime(df['open_dt'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude'])

# Filter only animal-related cases
animal_keywords = ['animal', 'dog', 'cat', 'wildlife', 'bite']
df = df[df['type'].str.lower().str.contains('|'.join(animal_keywords), na=False)]

# Start Dash app
app = dash.Dash(__name__)
app.title = "Boston Animal-Related 311 Dashboard"

# Layout
app.layout = html.Div([
    html.H1("\ud83d\udcca Boston Animal-Related 311 Explorer", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Case Type:"),
        dcc.Dropdown(
            options=[{"label": t, "value": t} for t in sorted(df['type'].dropna().unique())],
            value=None,
            id='type-filter',
            multi=True,
            placeholder="Filter by animal case type...",
        ),
    ], style={"width": "40%", "margin": "auto"}),

    dcc.Graph(id='case-time-series'),
    dcc.Graph(id='case-map'),
])

# Callbacks
@app.callback(
    [Output("case-time-series", "figure"),
     Output("case-map", "figure")],
    [Input("type-filter", "value")]
)
def update_dashboard(selected_types):
    filtered = df.copy()
    if selected_types:
        filtered = filtered[filtered['type'].isin(selected_types)]

    # Time series
    time_series = (
        filtered['open_dt'].dt.date
        .value_counts()
        .sort_index()
        .reset_index(name='count')
        .rename(columns={'index': 'open_dt'})
    )

    fig1 = px.line(time_series, x='open_dt', y='count',
                   title="Animal-Related Cases Over Time")

    # Map - color by 'type' for distinct colors per case type
    fig2 = px.scatter_mapbox(
        filtered,
        lat="latitude",
        lon="longitude",
        color="type",
        hover_name="case_title",
        zoom=11,
        mapbox_style="carto-positron",
        title="Map of Animal-Related Cases"
    )

    return fig1, fig2

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)