from App import dbc,html,dcc

def render_form_KNN():
    container =dbc.Container([
        dbc.Row([dbc.Col([html.H3("handel messing Values",className="text-capitalize fw-bold fs-1 ")],className="col-12")],className =" mt-5"),
        dbc.Row([dbc.Col([html.H3("1.Using KNN imputer",className="text-capitalize fw-medium fs-3 ")],className="col-12")],className =" mt-3"),
        html.P("use with K-Nearest Neighbors",className="fs-5"),
        dbc.Row([dbc.Col([html.H3("about K-Nearest Neighbors ",className="text-capitalize fw-normal fs-5 px-md-5 ps-3")],className="col-12")],className =" my-3"),   
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label("Nearest Neighbors : ",className="text-nowrap "),
                        dbc.Input(id='knn_neibhord_IMP',type="number",placeholder="default 2  ...",min=1 ,size='md')
                    ],className="d-flex gap-5 justify-content-center align-items-center"),
                ],className="col-12 d-flex justify-content-xl-center align-items-center justify-content-start p-xl-0 ps-5"),
            ]),
            dbc.Row([dbc.Col([dbc.Button("run",id="btn_run_KNNIM",className="text-capitalize fw-normal fs-5 w-25")],className="col-12 d-flex justify-content-center ")],className =" mt-5"),
        ],className="mb-5 "),
        dbc.Spinner([
             html.Div(id="result_KNN",children=[])
        ])  
    ])
    return container