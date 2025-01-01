from App import dbc,html,dcc

def render_form_optimizer():
    container =html.Div([
        dbc.Row([dbc.Col([html.H3("2. using optimiser algorithmes",className="text-capitalize fw-medium fs-3 ")],className="col-12")],className =" mt-3"),
        html.P("use with K-Neirest-Neihbord and MLP-Classifier ",className="fs-5 "),
        html.Div([
            dbc.Row([dbc.Col([html.H3("About optimization ",className="text-capitalize  fs-4 fw-100 ")],className="col-12")],className ="my-3"),
             dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label("Nan Values Withing : ",className="text-nowrap me-5"),
                        dcc.Dropdown(options=[
                            {"label":"Train dataset ","value":"train"},
                            {"label":"Test dataset ","value":"test"},
                        ],value="test",multi=False,clearable=False,id="_drpdwn_strtg",style={'width':'200px'})
                    ],className="d-flex ps-5 justify-content-xl-center  align-items-center  w-75 ")
                ],className='col-xl-4 col-12 mt-3 mt-xl-0  '),
                dbc.Col([
                    html.Div([
                        dbc.Label("Random seed: ",className="text-nowrap "),
                        dbc.Input(id='randomseed',type="number",placeholder="default 42  ...",min=1 ,size='md')
                    ],className="d-flex gap-5 justify-content-center align-items-center ps-xl-0 ps-5"),
                ],className="col-xl-4 col-12 mt-3 mt-xl-0  "),
                
                
            ],className ="my-3 justify-content-md-around justify-content-center flex-xl-row flex-col "),
            dbc.Row([dbc.Col([html.H3(" about Optimiser ",className="text-capitalize  fs-4 fw-400 ")],className="col-12")],className ="my-3"),
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
                        ],value="SMO",multi=False,clearable=False,id="_drpdwn_optmzr",style={'width':'200px'})
                    ],className="d-flex  justify-content-xl-center  align-items-center  ps-xl-0 ps-5 ")
                ],className='col-xl-4 col-12'),

                dbc.Col([
                    html.Div([
                        dbc.Label("Epocks : ",className="text-nowrap "),
                        dbc.Input(id='epocks',type="number",placeholder="default 2  ...",min=1 ,size='md')
                    ],className="d-flex gap-5 justify-content-center align-items-center ps-xl-0 ps-5"),
                ],className="col-xl-4 col-12 mt-3 mt-xl-0  "),
                dbc.Col([
                    html.Div([
                        dbc.Label("Poplation size : ",className="text-nowrap"),
                        dbc.Input(id='popsize',type="number",placeholder="default 3  ...",min=1,size='md' )
                    ],className="d-flex gap-5 justify-content-center align-items-center ps-xl-0 ps-5"),
                ],className="col-xl-4 col-12 mt-3 mt-xl-0 "),
                
                
                
            ],className ="my-3 justify-content-md-around justify-content-center   "),
            
            dbc.Row([dbc.Col([html.H3("about K-Nearest Neighbors ",className="text-capitalize fs-4 fw-400 ")],className="col-12")],className =" my-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label("Nearest Neighbors : ",className="text-nowrap "),
                        dbc.Input(id='knn_neirest_opt',type="number",placeholder="default 2  ...",min=1 ,size='md')
                    ],className="d-flex gap-5 justify-content-center align-items-center"),
                ],className="col-12 d-flex justify-content-xl-center align-items-center justify-content-start p-xl-0 ps-5"),
            ]),
            dbc.Row([dbc.Col([html.H3("about MLP-Classifier ",className="text-capitalize fs-4 fw-400 ")],className="col-12")],className =" my-3"),
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

def render_Graph(fig):
    container =html.Div([
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

def render_table(result:dict,warnings:dict):
    knn_std=result.get('with_std').get('KNN')
    knn_without_std=result.get('without_std').get('KNN')
    mlp_std=result.get('with_std').get('MLP')
    mlp_without_std=result.get('without_std').get('MLP')
    container =html.Div([
        dbc.Alert([ html.Span(i,className=' p-3 text-capitalise')  for i in warnings.values()],color='warning',dismissable=True,className="mt-5 d-flex flex-column rounded "),
        dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("",rowSpan=2,className="text-center"),
                    html.Th("K-Neirest-Neihbord",colSpan=2,className="text-center"),
                    html.Th("MLP-Classifier",colSpan=2,className="text-center")
                ]),
                html.Tr([
                    html.Th("with standardisation",className="text-center"),
                    html.Th("without standardisation",className="text-center"),
                    html.Th("with standardisation",className="text-center"),
                    html.Th("without standardisation",className="text-center"),
                ]),
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("accuracy",className="fs-5 fw-bold text-center"),
                    html.Td(f"{knn_std.get('accuracy').get('Fitness')}%",className="text-center"),
                    html.Td(f"{knn_without_std.get('accuracy').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_std.get('accuracy').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_without_std.get('accuracy').get('Fitness')}%",className="text-center"),
                ]),
                html.Tr([
                    html.Td("f1_score",className="fs-5 fw-bold text-center"),
                    html.Td(f"{knn_std.get('f1_score').get('Fitness')}%",className="text-center"),
                    html.Td(f"{knn_without_std.get('f1_score').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_std.get('f1_score').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_without_std.get('f1_score').get('Fitness')}%",className="text-center"),
                ]),
                html.Tr([
                    html.Td("recall",className="fs-5 fw-bold text-center"),
                    html.Td(f"{knn_std.get('recall').get('Fitness')}%",className="text-center"),
                    html.Td(f"{knn_without_std.get('recall').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_std.get('recall').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_without_std.get('recall').get('Fitness')}%",className="text-center"),
                ]),
                html.Tr([
                    html.Td("precision",className="fs-5 fw-bold text-center"),
                    html.Td(f"{knn_std.get('precision').get('Fitness')}%",className="text-center"),
                    html.Td(f"{knn_without_std.get('precision').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_std.get('precision').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_without_std.get('precision').get('Fitness')}%",className="text-center"),
                ]),
            ])
            
        ],responsive=True,hover=True )
    ],className="px-5 py-3 my-5")
    return container




