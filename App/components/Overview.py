from App import dbc,html

def overview(result:dict):
    col_nan = result.get("features_nan") 
    render = f"{result.get('messing_values')}  ( {' , '.join([f'{i} : {col_nan.get(i)}' for i in col_nan.keys()])} )"
    
    
    
    container =dbc.Container([
        dbc.Row([dbc.Col([html.H3("Overview about dataset",className="text-capitalize fw-bold fs-1 ")],className="col-12")],className =" mt-5"),
        html.Div([
            dbc.ListGroup([
                dbc.ListGroupItem([html.Span("sambles  :",className="fs-4 text-capitalize"),result.get("Sambles")],className="fs-5 text-uppercase"),
           
                dbc.ListGroupItem([html.Span("features :",className="fs-4 text-capitalize ") ,' , '.join(result.get("features"))] 
                                  ,className="fs-5 text-uppercase"),
                dbc.ListGroupItem([html.Span("target :",className="fs-4 text-capitalize"),result.get("target")],className="fs-5 text-uppercase"),
                dbc.ListGroupItem([html.Span("messing values :",className="fs-4 text-capitalize"),
                                   render ,
                                   ],className="fs-5 text-uppercase"),
            ],flush=True),
            
        ],className="px-5 mt-4")
    ])
    
    return container