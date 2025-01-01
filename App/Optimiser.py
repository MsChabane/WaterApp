
from App.Algorithms import PO,SMO,HHO,MRFO,RUN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class Optimiser:
    def __init__(self,data):
        self.data=data
        self.messing_values=np.isnan(data).sum().sum()
        self.col_messing_values_indecies = np.argwhere(data.isna().any(axis=0)).ravel()
        self.minmax=[(data.iloc[:,i].min(),data.iloc[:,i].max()) for i in self.col_messing_values_indecies ]
        self.not_cleaned_data=data.iloc[np.argwhere(data.isna().any(axis=1)).ravel()]
        self.cleaned_data=data.dropna()
        
        self.standardscaler = StandardScaler()
        self.metrics={"accuracy":accuracy_score,"f1_score":f1_score,"recall":recall_score,"precision":precision_score}
        self.optimiser={"PO":PO,"HHO":HHO,"MRFO":MRFO,"SMO":SMO,"RUN":RUN}
        self.result={}



    def _impute_values(self,df,solution):
        count=0
        for idx in range(self.col_messing_values_indecies.shape[0]):
            mask = df.iloc[:,self.col_messing_values_indecies[idx]].isna()
            imputed_values=(self.minmax[idx][0] + solution[count:count+mask.sum()] % (self.minmax[idx][1] -self.minmax[idx][0] )).astype("float16")
            df.iloc[np.argwhere(mask).ravel(),self.col_messing_values_indecies[idx]]=imputed_values
            count+=mask.sum()
        return df

    def solve(self,algorithm:str,num_epk=5,pop_size:int=5,n_neighbors=5,neurons_num:int=10,strategy="test",random_seed=0):
        if algorithm not in self.optimiser.keys():
            raise Exception("optimiser not found")
        self.random_seed=random_seed
        self.models=  {
            "KNN":KNeighborsClassifier(n_neighbors=n_neighbors),
            "MLP":MLPClassifier(random_state=self.random_seed, max_iter=400,hidden_layer_sizes=(neurons_num))
        }
        op_model=self.optimiser.get(algorithm)(pop_size,num_epk,-5000,5000,self.messing_values)
        self.train,self.test=train_test_split(self.cleaned_data,test_size=0.2,random_state=self.random_seed)
        (self._solving_nan_in_test if strategy=="test" else self._solving_nan_in_train )(op_model)
        

    def _solving_nan_in_test(self,optimiser):
        self.concatenation = pd.concat((self.test,self.not_cleaned_data))
        result_metric={}
        result_model={}
        for method in ["without_std","with_std"]:
          self._make_standardisation=True if method=="with_std" else False
          print("with Standardisation" if self._make_standardisation else "without Standardisation")
          if self._make_standardisation:
              self.train.iloc[:,:-1]=self.standardscaler.fit_transform(self.train.iloc[:,:-1])
          for model in self.models.keys():
              self.current_model=model
              print(f"Using {self.current_model}")
              self.models.get(self.current_model).fit(self.train.iloc[:,:-1],self.train.iloc[:,-1])
              for metric in self.metrics.keys():
                  self.current_metric=metric
                  print(f"      >>> metric used {self.current_metric}")
                  result = optimiser.solve(self._obj_func_nan_with_test)
                  result_metric[f"{self.current_metric}"]=self._make_result(result)
              result_model[model]=result_metric
              result_metric={}
          self.result[method]=result_model 
          result_model={}  
     
    def _solving_nan_in_train(self,optimiser):
        self.concatenation = pd.concat((self.train,self.not_cleaned_data))
        result_metric={}
        result_model={}
        for method in ["without_std","with_std"]:
          self._make_standardisation=True if method=="with_std" else False
          print("with Standardisation" if self._make_standardisation else "without Standardisation")
          if self._make_standardisation:
              self.test.iloc[:,:-1]=self.standardscaler.fit_transform(self.test.iloc[:,:-1])
          for model in self.models.keys():
              self.current_model=model
              print(f"Using {self.current_model}")
              for metric in self.metrics.keys():
                  self.current_metric=metric
                  print(f"      >>> metric used {self.current_metric}")
                  result = optimiser.solve(self._obj_func_nan_with_train)
                  result_metric[f"{self.current_metric}"]=self._make_result(result)
              result_model[model]=result_metric
              result_metric={}
          self.result[method]=result_model 
          result_model={}      

    


    def _obj_func_nan_with_train(self,solution):
        filled_data=self._impute_values(self.concatenation.copy(),solution)
        if self._make_standardisation:
            filled_data.iloc[:,:-1] =  self.standardscaler.fit_transform(filled_data.iloc[:,:-1])
        self.models.get(self.current_model).fit(filled_data.iloc[:,:-1],filled_data.iloc[:,-1])
        return 1-self.metrics.get(self.current_metric)(self.test.iloc[:,-1],self.models.get(self.current_model).predict(self.test.iloc[:,:-1]))


    def _obj_func_nan_with_test(self,solution):
        filled_data=self._impute_values(self.concatenation.copy(),solution)
        if self._make_standardisation:
            filled_data.iloc[:,:-1] =  self.standardscaler.fit_transform(filled_data.iloc[:,:-1])
        return 1-self.metrics.get(self.current_metric)(filled_data.iloc[:,-1],self.models.get(self.current_model).predict(filled_data.iloc[:,:-1]))

    def _make_result(self,output):
        return {
            "Fitness":round(1-output[1],3)*100,"Solution":output[0],"curve":1-output[2]
        }

    def _get_chemin_max_fitness(self,metric):
        max_vlaue=0
        chemin=""
        for method in self.result.keys():
            for model in self.result[method].keys():
                value=self.result.get(method).get(model).get(metric).get("Fitness")
                if value > max_vlaue:
                    max_vlaue=value
                    chemin=f"{method}-{model}"
        return chemin

    def get_imputed_dataset(self,metric):
        method,model =self._get_chemin_max_fitness(metric).split("-")
        return self._impute_values(self.data.copy(),self.result.get(method).get(model).get(metric).get("Solution"))




