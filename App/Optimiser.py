
from App.Algorithms import PO,SMO,HHO,MRFO,RUN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

class Optimiser:
    def __init__(self,data,random_seed):
        self.data=data
        self.messing_values=np.isnan(data).sum().sum()
        self.col_messing_values_indecies = np.argwhere(data.isna().any(axis=0)).ravel()
        self.minmax=[(data.iloc[:,i].min(),data.iloc[:,i].max()) for i in self.col_messing_values_indecies ]
        self.not_cleaned_data=data.iloc[np.argwhere(data.isna().any(axis=1)).ravel()]
        self.cleaned_data=data.dropna()
        self.random_seed=random_seed
        self.standardscaler = StandardScaler()
        
        self.metrics={"accuracy":accuracy_score,"f1_score":f1_score,"recall":recall_score,"precision":precision_score}
        self.metrics_names=list(self.metrics.keys())
        self.models={"PO":PO,"HHO":HHO,"MRFO":MRFO,"SMO":SMO,"RUN":RUN}
        self.result={}

    
    
    def _impute_values(self,df,solution):
        count=0
        for idx in range(self.col_messing_values_indecies.shape[0]):
            mask = df.iloc[:,self.col_messing_values_indecies[idx]].isna()
            imputed_values=(self.minmax[idx][0] + solution[count:count+mask.sum()] % (self.minmax[idx][1] -self.minmax[idx][0] )).astype("float16")
            df.iloc[np.argwhere(mask).ravel(),self.col_messing_values_indecies[idx]]=imputed_values
            count+=mask.sum()
        return df

    def solve(self,model:str,num_epk=5,pop_size:int=5,n_neighbors=5,neurons_num:int=10):
        if model not in self.models.keys():
            raise Exception("model not found")
        self.knn=KNeighborsClassifier(n_neighbors=n_neighbors)
        self.mlp=MLPClassifier(random_state=self.random_seed, max_iter=2000,hidden_layer_sizes=(neurons_num))
        op_model=self.models.get(model)(pop_size,num_epk,-5000,5000,self.messing_values)
        self.train,self.test=train_test_split(self.cleaned_data,test_size=0.2,random_state=self.random_seed)
        self.concatenation = pd.concat((self.test,self.not_cleaned_data))
        self._with_knn(op_model)
        self._with_mlp(op_model)

    def _with_knn(self,model):
        dic={}
        dic_standard={}
        print('Using KNN')
        print(">>> without standardisation ")
        self.knn.fit(self.train.iloc[:,:-1],self.train.iloc[:,-1])
        self.current_metric=""
        for i in range(self.metrics.__len__()):
            self.current_metric=self.metrics_names[i]
            print(f"      >>> metric used {self.current_metric}")
            gbest=model.solve(self._obj_func_KNN_False)
            dic[f"{self.current_metric}"]=self._make_result( gbest)

        dic_standard["without_std"]=dic
        dic={}
        print(">>> with standardisation ")
        self.knn.fit(pd.DataFrame(self.standardscaler.fit_transform(self.train.iloc[:,:-1]),columns=self.train.columns[:-1]),self.train.iloc[:,-1])

        for i in range(self.metrics.__len__()):
            self.current_metric=self.metrics_names[i]
            print(f"      >>> metric used {self.current_metric}")
            gbest=model.solve(self._obj_func_KNN_True)
            dic[f"{self.current_metric}"]=self._make_result( gbest)
        dic_standard["with_std"]=dic
        self.result["KNN"]=dic_standard



    def _obj_func_KNN_True(self,solution):
        filled_data=self._impute_values(self.concatenation.copy(),solution)
        filled_data.iloc[:,:-1] =  self.standardscaler.fit_transform(filled_data.iloc[:,:-1])
        return 1-self.metrics.get(self.current_metric)(filled_data.iloc[:,-1],self.knn.predict(filled_data.iloc[:,:-1]))

    def _obj_func_KNN_False(self,solution):
        filled_data=self._impute_values(self.concatenation.copy(),solution)
        return 1-self.metrics.get(self.current_metric)(filled_data.iloc[:,-1],self.knn.predict(filled_data.iloc[:,:-1]))

    def _with_mlp(self,model):
        dic={}
        dic_standard={}
        print('Using MLP ')
        print(">>> without standardisation ")

        self.mlp.fit(self.train.iloc[:,:-1],self.train.iloc[:,-1])
        self.mlp_coef_Without_STD=self.mlp.coefs_
        for i in range(self.metrics.__len__()):
            self.current_metric=self.metrics_names[i]
            print(f"      >>> metric used {self.current_metric}")
            gbest=model.solve(self._obj_func_MLP_False)
            dic[f"{self.current_metric}"]=self._make_result( gbest)
        dic_standard["without_std"]=dic
        self.mlp.fit(pd.DataFrame(self.standardscaler.fit_transform(self.train.iloc[:,:-1]),columns=self.train.columns[:-1]),self.train.iloc[:,-1])
        self.mlp_coef_With_STD=self.mlp.coefs_
        print(">>> with standardisation ")
        dic={}
        for i in range(self.metrics.__len__()):
            self.current_metric=self.metrics_names[i]
            print(f"      >>> metric used {self.current_metric}")
            gbest=model.solve(self._obj_func_MLP_True)
            dic[f"{self.current_metric}"]=self._make_result( gbest)
        dic_standard["with_std"]=dic
        self.result["MLP"]=dic_standard

    def _obj_func_MLP_True(self,solution):
        filled_data=self._impute_values(self.concatenation.copy(),solution)
        filled_data.iloc[:,:-1] =  self.standardscaler.fit_transform(filled_data.iloc[:,:-1])
        self.mlp.coefs_=self.mlp_coef_With_STD
        return 1-self.metrics.get(self.current_metric)(filled_data.iloc[:,-1],self.mlp.predict(filled_data.iloc[:,:-1]))

    def _obj_func_MLP_False(self,solution):
        filled_data=self._impute_values(self.concatenation.copy(),solution)
        self.mlp.coefs_=self.mlp_coef_Without_STD
        return 1-self.metrics.get(self.current_metric)(filled_data.iloc[:,-1],self.mlp.predict(filled_data.iloc[:,:-1]))
    def _make_result(self,output):
        return {
            "Fitness":np.round(1-output[1],3),"Solution":output[0],"curve":1-output[2]
        }

    def _get_chemin_max_fitness(self,metric):
        max_vlaue=0
        chemin=""
        for model in self.result.keys():
            for method in self.result[model].keys():
                value=self.result.get(model).get(method).get(metric).get("Fitness")
                if value > max_vlaue:
                    max_vlaue=value
                    chemin=f"{model}-{method}"
        return chemin

    def get_imputed_dataset(self,metric):
        model,method =self._get_chemin_max_fitness(metric).split("-")
        return self._impute_values(self.data.copy(),self.result.get(model).get(method).get(metric).get("Solution"))
        


