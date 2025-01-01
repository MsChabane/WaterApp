import dash 
import dash_bootstrap_components as dbc
from dash import  html,dcc
from App.components.header import render_header
from App.components.Optimiser_Section import render_form_optimizer
from App.components.KnnImputer_Section import render_form_KNN

app=dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,],prevent_initial_callbacks=True)
app.title="Water Potability "

app.layout=html.Div([
    render_header(),
    dbc.Container(id="container_Overview",children=[]),
    dbc.Container(id="container_KNN",children=[]),
    dbc.Container(id="container_optimiser",children=[])
    
])

import App.Actions
