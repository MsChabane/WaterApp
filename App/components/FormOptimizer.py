from App import dbc,html,dcc


def render_form_optimizer():
    container =dbc.Container([
        dbc.Row([dbc.Col([html.H3("2. using optimiser algorithmes",className="text-capitalize fw-medium fs-3 ")],className="col-12")],className =" mt-3"),
        html.P("use with K-Neirest-Neihbord and MLP-Classifier ",className="fs-5 "),
        html.Div([
            dbc.Row([dbc.Col([html.H3("About optimization ",className="text-capitalize fw-normal fs-5 ")],className="col-12")],className ="my-3"),
             dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label("Optimizer : ",className="text-nowrap me-5"),
                        dcc.Dropdown(options=[
                            {"label":"Harris Hawks Optimizer ","value":"HHO"},
                            {"label":"Parrot Optimizer","value":"PO"},
                            {"label":"Runge Kutta Optimizer","value":"RUN"},
                            {"label":"Slime Mould Optimizer","value":"SMO"},
                            {"label":"Manta Ray Foraging Optimizer","value":"MRFO"},
                        ],value="SMO",multi=False,placeholder="Select an optemizer",clearable=False,id="_drpdwn_optmzr",style={'width':'70%'})
                    ],className="d-flex ps-5 justify-content-xl-center  align-items-center  w-75 ")
                ],className='col-12 d-flex justify-content-start ps-3 ps-xl-0 align-items-center ')
                
                
            ],className ="my-3 justify-content-md-around justify-content-center  flex-xl-row flex-column "),
            dbc.Row([dbc.Col([html.H3(" about Optimiser ",className="text-capitalize fw-normal fs-5 ")],className="col-12")],className ="my-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label("Epocks : ",className="text-nowrap "),
                        dbc.Input(id='epocks',type="number",placeholder="default 2  ...",min=1 ,size='md')
                    ],className="d-flex gap-5 justify-content-center align-items-center ps-xl-0 ps-5"),
                ],className="col-xl-4 col-12  "),
                dbc.Col([
                    html.Div([
                        dbc.Label("Poplation size : ",className="text-nowrap"),
                        dbc.Input(id='popsize',type="number",placeholder="default 3  ...",min=1,size='md' )
                    ],className="d-flex gap-5 justify-content-center align-items-center ps-xl-0 ps-5"),
                ],className="col-xl-4 col-12 mt-3 mt-xl-0 "),
            ],className ="my-3 justify-content-md-around justify-content-center   "),
            
            dbc.Row([dbc.Col([html.H3("about K-Nearest Neighbors ",className="text-capitalize fw-normal fs-5 ")],className="col-12")],className =" my-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label("Nearest Neighbors : ",className="text-nowrap "),
                        dbc.Input(id='knn_neirest_opt',type="number",placeholder="default 2  ...",min=1 ,size='md')
                    ],className="d-flex gap-5 justify-content-center align-items-center"),
                ],className="col-12 d-flex justify-content-xl-center align-items-center justify-content-start p-xl-0 ps-5"),
            ]),
            dbc.Row([dbc.Col([html.H3("about MLP-Classifier ",className="text-capitalize fw-normal fs-5 ")],className="col-12")],className =" my-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label("Hidden layer size  : ",className="text-nowrap "),
                        dbc.Input(id='mpl_hlsize',type="number",placeholder="default 10  ...",min=1 ,size='md')
                    ],className="d-flex gap-5 justify-content-center align-items-center"),
                ],className="col-12 d-flex justify-content-xl-center align-items-center justify-content-start p-xl-0 ps-5"),
            ]),
            dbc.Row([dbc.Col([dbc.Button("run",id="btn_run_po",className="text-capitalize fw-normal fs-5 w-25")],className="col-12 d-flex justify-content-center ")],className =" my-5"),
            ],className="px-md-5  px-3")
        ,
        dbc.Spinner([html.Div(id="result_Optemizer",children=[])],fullscreen=True)
        
    ])
    
    return container