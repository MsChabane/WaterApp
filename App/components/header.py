from App import dbc,html,dcc

def render_header():
    header =html.Div( [
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H3("Water Potability",className="text-center text-primary fs-1 mx-3 fw-bold")
                ]), 
            ],className="mb-5 mt-3"),
            dbc.Row([
                dbc.Col([
                    html.Img(src="./../assets/water.jfif",className="w-100 h-100 rounded ")
                    
                    ],className="col-12 col-md-4   "),

                dbc.Col([
                    html.P("A Potable water is suitable for human consumption",className="text-start fs-3 text-capitalize")
                ],className="col-12 col-md-8 d-flex align-items-md-center ")
                
            ],className="p-5 d-flex flex-column-reverse flex-md-row-reverse justify-content-md-around mb-5 mx-3"),
            dbc.Row([
                dbc.Col([
                    html.P([html.Span("""water potability dataset""",className="fw-bold text-decoration fs-2"),""" is commonly used in machine learning to predict whether a sample of water is potable (safe for drinking)
                        based on various physical and chimical attributes."""],className="fs-3 text-capitalize text-break lh-lg px-5")
                ],className="col-12 my-3")
            ])
        ],className="min-vh-100 d-flex flex-column justify-content-center")
        ,
        dbc.Row([
            dbc.Col([
                dbc.Button("Load and analyse the dataset",className="px-3 py-2 fs-4  fw-bold bg-outline-primary",id="btn_load")
            ],className="col-12 d-flex justify-content-center align-items-center")
        ],className="pb-5"),

    ],className="")
    return header
