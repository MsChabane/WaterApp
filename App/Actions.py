from dash import Output,Input,no_update,State
from App import app,dbc,html
from App.Optimiser import Optimiser
from App.components.Overview import overview
from App.components.ResultKNNimputer import render_result_Knnimputer
from App.components.FormOptimizer import render_form_optimizer
from App.components.Table import render_table
from App.components.fitness_graph import render_Graph
from App.components.Form_KNNImputer import render_form_KNN
from App.components.EDA import render_form_EDA_Section,render_result_EDA

import numpy as np 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn .impute import KNNImputer
from sklearn .neighbors import KNeighborsClassifier
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


df_filled=None
df=None
optimiser=None

@app.callback([
    Output("btn_load","disabled"),
    Output("container_Overview","children"),
    Output("container_KNN","children"),
    Output("container_optimiser","children"),
    Input("btn_load","n_clicks")],prevent_initial_call=True)
def handel_load_click(n):
    global df 
    global optimiser
    if df is None :
        df = pd.read_csv("./App/assets/waterpotability.csv")
        optimiser =Optimiser(data=df,random_seed=0)
    _check_nan= np.isnan(df).sum() 
    _to_overview={
        "Sambles":df.shape[0],
        "features":list(df.columns[:-1]),
        "target":df.columns[-1],
        "messing_values":_check_nan.sum(),
        "features_nan":_check_nan[_check_nan !=0].to_dict()
    }
    return True,[overview(_to_overview)],[render_form_KNN()],[render_form_optimizer()]

@app.callback([
    Output("result_KNN","children"),
    Input("btn_run_KNNIM","n_clicks"),
    State("knn_neibhord_IMP","value"),
],prevent_initial_call=True)
def handel_KNN_Imputer(n,knn_neibhord):
    warning=False
    global df 
    if  knn_neibhord ==None or knn_neibhord > df.shape[0]:
        warning=True
        #alert = dbc.Alert("Neirest neibhord must be smaller than sambels ,By default used 2", color="warning",dismissable=True)
        knn_neibhord=2
    train,test=train_test_split(KNNImputer(n_neighbors=knn_neibhord).fit_transform(df))
    knn_classifier = KNeighborsClassifier(n_neighbors=knn_neibhord).fit(train[:,:-1],train[:,-1])
    prediction = knn_classifier.predict(test[:,:-1])
    _to_knn_impute_result={
        "accuracy":np.round(accuracy_score(test[:,-1], prediction)*100,2),
        "f1_score":np.round(f1_score(test[:,-1], prediction)*100,2),
        "recall":np.round(recall_score(test[:,-1], prediction)*100,2),
        "precision":np.round(precision_score(test[:,-1], prediction)*100,2)}
    
    return [
            render_result_Knnimputer(_to_knn_impute_result,make_warning=warning)
            ]


def render_warnings(e,p,n,h):
    global df
    warnings={}
    if e ==None :
        warnings["epocks"]="Used the default value of 2 for epocks "
    elif e >1000:
        warnings["epocks"]="too large value for epocks, used default  value of 2"
    if p ==None :
        warnings["popsize"]="Used the default value of 2 for Populatoin size "
    elif p >50:
        warnings["popsize"]="too large value for Population size, Used the default value of 2"
    if n ==None :
        warnings["neibhord"]="Used the default value of 2 for K-neirest neibhord "
    elif n >df.shape[0]:
        warnings["neibhord"]="too large value for K-neirest neibhord ,Used default value of 2 "
    if h ==None :
        warnings["hiddenSize"]="Used the default value of 10 for Mlp hidden layer size "
    elif h >300:
        warnings["hiddenSize"]="too large value for Mlp hidden layer size,Used default value of 10 "
    return warnings

def make_graph(method,metric):
    global optimiser
    fig = go.Figure(
        data=[
           go.Scatter(y=optimiser.result.get("KNN").get(method).get(metric).get("curve"),name="KN-Neighbors",mode="lines+markers") ,
           go.Scatter(y=optimiser.result.get("MLP").get(method).get(metric).get("curve"),name="MLP-Classifier",mode="lines+markers"), 
        ],
        layout={
            'title':{'text':f"distrebution of {metric} {'with standardisation' if method=='with_std' else ''}"
                     ,'x':0.5 },'xaxis':{'title':'#epocks'},'yaxis':{'title':metric}
        }
    )
    
    return fig
 

@app.callback([
    Output("result_Optemizer","children"),
    Input("btn_run_po","n_clicks"),
    State("_drpdwn_optmzr","value"),
    State("epocks","value"),
    State("popsize","value"),
    State("knn_neirest_opt","value"),
    State("mpl_hlsize","value"),
    
])
def handel_Optimiser(n,model,epocks,popsize,n_neighbors,hidden_layer_size):
    global optimiser
    warnings =render_warnings(epocks,popsize,n_neighbors,hidden_layer_size)
    epocks = 2 if warnings.get("epocks") else epocks
    popsize = 2 if warnings.get("popsize") else popsize
    n_neighbors = 2 if warnings.get("neibhord") else n_neighbors
    hidden_layer_size = 10 if warnings.get("hiddenSize") else hidden_layer_size
    optimiser.solve(model=model,num_epk=epocks,pop_size=popsize,n_neighbors=n_neighbors,neurons_num=hidden_layer_size)
    
    figure = make_graph("with_std","accuracy")
    
    
    return [[
        dbc.Alert([ html.Span(i,className=' p-3 text-capitalise')  for i in warnings.values()],color='warning',dismissable=True,className="mt-5 d-flex flex-column rounded"),
        render_table(optimiser.result),
        render_Graph(figure),
        render_form_EDA_Section()
       
       ]]
   

    
@app.callback([
        Output("fitness_graph","figure"),
        Input("btn_show_graph","n_clicks"),
        State("_drpdwn_method","value"),
        State("_drpdwn_metric","value")
],prevent_initial_call=True)
def handel_graph_change(n,method,metric):
    figure = make_graph(method,metric)
    return figure,
   
   
def _make_graphs(colName): 
    global df_filled
    boxplot = px.box(data_frame=df_filled,x=colName)
    histplot = px.histogram(data_frame=df_filled,x=colName)
    return boxplot,histplot
    
    
@app.callback([
    Output("eda_result","children"),
    Input("btn_fill_data","n_clicks"),
    State("_drpdwn_metric","value")
])   
def fill_data(n,metric):
    global df_filled
    global optimiser
    df_filled =optimiser.get_imputed_dataset(metric)
    options =[{"label":df_filled.columns[idx] ,"value":df_filled.columns[idx] } for idx in optimiser.col_messing_values_indecies]
    value=options[0].get('value')
    box,hist=_make_graphs(value)
    return [
        [
            render_result_EDA(options=options,value=value,boxfig=box,histfig=hist)
        ]
    ]
    
    
@app.callback([
    Output("boxplot",'figure'),
    Output("histogramplot",'figure'),
    Input("btn_show_columns_graphs","n_clicks"),
    State("_drpdwn_column","value")
])
def handel_graphs(n,colName):
    box,hist=_make_graphs(colName)
    return [
        box,
        hist
    ]
    
    
    
    
    
    
