import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import hddm
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from kabuki.analyze import gelman_rubin

#set number of synthetic subjects and amount of trial to generate per subject
subjects = 40
trials = 60

#create array with all combinations of three values for a, t, scaler and alpha
x = np.array([(a, t, scaler, alpha)
              for a in np.linspace(1.5,2.5,num=3) 
              for t in np.linspace(0.3,0.5,num=3) 
              for scaler in np.linspace(1.5,3,num=3) 
              for alpha in np.linspace(0.15,0.45,num=3)
              ])
f = pd.DataFrame(data=x,columns=['a','t', 'scaler', 'alpha'])

#generate data, estimate model and save traces and convergence info for all combinations of models
#this will take a while, so sending these as jobs on a cluster is recommended
for i in range(f.shape[0]):
  data = hddm.generate.gen_rand_rlddm_data(a=f.a[i],alpha=f.alpha[i],scaler=f.scaler[i],t=f.t[i],size=trials,subjs=subjects,p_upper=0.75,p_lower=0.25)
  data['q_init'] = 0.5
  models = []
  for a in range(3):  
    m = hddm.HDDMrl(data=data)
    m.sample(3000,burn=1500,dbname='traces.db',db='pickle')
    models.append(m)
  gelman = gelman_rubin(models)
  gelman = pd.DataFrame.from_dict(gelman,orient='index')
  convergencename = 'convergence_model%s.csv'%(i)
  gelman.to_csv(convergencename)
  traces = m.get_traces()
  filename = 'traces_model%s.csv'%(i)
  traces.to_csv(filename)

