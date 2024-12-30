from App import dbc,html


def render_result_Knnimputer(result:dict,make_warning=False):
    
    container = dbc.Container([
        make_warning and dbc.Alert("Used the default value of 2 for K-nearest neighbors", color="warning",dismissable=True),
        html.Div([
            html.Label("Accuracy Score",className="fs-5"),
            dbc.Progress(label=f"{result.get('accuracy')}%",value=result.get('accuracy'),className="mb-3  fs-6") ,
            html.Label("F1_Score",className="fs-5"),
            dbc.Progress(label=f"{result.get('f1_score')}%",value=result.get('f1_score'),className="mb-3  fs-6") ,
            html.Label("Recall Score",className="fs-5"),
            dbc.Progress(label=f"{result.get('recall')}%",value=result.get('recall'),className="mb-3  fs-6") ,
            html.Label("Precision Score",className="fs-5"),
            dbc.Progress(label=f"{result.get('precision')}%",value=result.get('precision'),className="mb-3  fs-6") ,
            ],className="px-5 py-3")
    ])
    return container
