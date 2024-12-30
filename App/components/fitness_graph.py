from App import dbc,html,dcc


def render_Graph(fig):
    container =dbc.Container([
        dcc.Graph(id='fitness_graph',figure=fig),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(options=[
                    {"label":"Without standardisation","value":"without_std"},
                    {"label":"With standardisation","value":"with_std"}
                ],value="with_std",multi=False,placeholder="Select the method",clearable=False,id="_drpdwn_method")
            ],className="col-md-4 col-8  "),
            dbc.Col([
                dcc.Dropdown(options=[
                    {"label":"Accuracy Score","value":"accuracy"},
                    {"label":"F1 Score","value":"f1_score"},
                    {"label":"Recall Score","value":"recall"},
                    {"label":"Precision Score","value":"precision"},
                ],value="accuracy",multi=False,placeholder="Select the metric",clearable=False,id="_drpdwn_metric")
            ],className="col-md-4 col-8  mt-md-0 mt-5"),
        ],className="my-5 d-flex justify-content-around align-items-center flex-md-row flex-column "),
        dbc.Row([dbc.Col([dbc.Button("Show the graph",id="btn_show_graph",className="text-capitalize fw-normal fs-5 w-25")],className="col-12 d-flex justify-content-center ")],className =" my-5"),
    ])

    return container




