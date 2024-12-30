import dash 
import dash_bootstrap_components as dbc
from dash import  html,dcc
from App.components.header import header

app=dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,],prevent_initial_callbacks=True)
app.title="Water Potability "

app.layout=html.Div([
    header,
    dbc.Container(id="container_Overview",children=[]),
    dbc.Container(id="container_KNN",children=[]),
    dbc.Container(id="container_optimiser",children=[])
    
])

import App.Actions
