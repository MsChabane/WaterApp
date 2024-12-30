from  App import dcc,dbc,html


def render_form_EDA_Section():
    container  = dbc.Container(
        [
           dbc.Row([dbc.Col([html.H3("Exploratory data analysis",className="text-capitalize fw-bold fs-1 ")],className="col-12")],className =" mt-5"), 
           html.P("After identifying the optimal solution for each metric ,apply these solutions to populate the dataset, and proceed with data exploration .",className="fs-5 mt-2 text-wrap"),
            html.Div([
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select the metric : "),
                        dcc.Dropdown(options=[
                                {"label":"Accracy Score","value":"accuracy"},
                                {"label":"F1 Score","value":"f1_score"},
                                {"label":"Recall Score","value":"recall"},
                                {"label":"Precision Score","value":"precision"},
                            ],value="accuracy",multi=False,clearable=False,id="_drpdwn_metric",style={'width':'200px'})
                    ],className='col-12 d-flex  gap-xl-3 gap-0 px-5 flex-column flex-xl-row align-items-xl-center justify-content-xl-center align-itens-start'),
                ],className='mb-3 '),
                dbc.Row([dbc.Col([dbc.Button("populate the dataset",id="btn_fill_data",className="text-capitalize fw-normal fs-5")],className="col-12 d-flex justify-content-xl-center justify-content-start ")],className ="mt-3 px-5"),
                ],className=' py-3  '),
                html.Div(children=[],id="eda_result")
        ]
       ,className='mb-4' 
    )
    return container


def render_result_EDA (options,value,boxfig,histfig):
    container=dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='boxplot',figure=boxfig)
            ],className='col-xl-5  col-10 d-flex jutify-content-center '),
            dbc.Col([
                dcc.Graph(id='histogramplot',figure=histfig)
            ],className='col-xl-5  col-10 d-flex jutify-content-center ')
        ])
        ,dbc.Row([
            dbc.Col([
                 dbc.Label("Select the feature : "),
                    dcc.Dropdown(options=options,value=value,multi=False,clearable=False,id="_drpdwn_column",style={'width':'200px'})
            ],className=' gap-3 px-5 d-flex justify-content-xl-center justify-content-start mb-3'),
          dbc.Col([dbc.Button("show graphs",id="btn_show_columns_graphs",className="text-capitalize fw-normal fs-5 ")],className="col-12 d-flex justify-content-xl-center justify-content-start px-5")  
        ])
    ])
    return container

