import numpy as np
import pandas as pd

models = ["BART","T5","PEGASUS","GPT2","XLNet"]

values = np.array([
    [0.89,7,8,0.88],
    [0.91,6,7,0.90],
    [0.92,5,6,0.91],
    [0.87,9,9,0.86],
    [0.90,8,7,0.89]
],dtype=float)

weights = np.array([1,1,1,1],dtype=float)

impacts = np.array([1,1,-1,1],dtype=int)

norm = values/np.sqrt((values**2).sum(axis=0))

weighted = norm*weights

best = np.where(impacts==1,weighted.max(axis=0),weighted.min(axis=0))
worst = np.where(impacts==1,weighted.min(axis=0),weighted.max(axis=0))

dpos = np.sqrt(((weighted-best)**2).sum(axis=1))
dneg = np.sqrt(((weighted-worst)**2).sum(axis=1))

score = dneg/(dpos+dneg)

order = score.argsort()[::-1]

rank = np.empty_like(order)
rank[order] = np.arange(1,len(score)+1)

output = pd.DataFrame({
    "Model":models,
    "Accuracy":values[:,0],
    "Speed":values[:,1],
    "Memory":values[:,2],
    "F1":values[:,3],
    "Score":score,
    "Rank":rank
})

output.to_excel("result.xlsx",index=False)

print(output.to_string(index=False))
