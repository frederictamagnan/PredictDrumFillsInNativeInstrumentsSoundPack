
from scipy import stats
import numpy as np
filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/'
# filename='FillsExtractedSupervised_c.npz'
# filename='validation.npz'
filename='FillsExtractedRuleBased_c.npz'
val=np.load(filepath+filename)
val=dict(val)
val=val['track_array']

val=(val>0)*1
print(val.shape)

def get_chi_square_statistic(observed, expected):
    return sum([((o - e)**2)/e for o, e in zip(observed, expected)])
def testchi2(test):

    rp=test[:,0,:,:].reshape((-1,9))
    f=test[:,1,:,:].reshape((-1,9))

    list_rp = []
    list_f = []

    for i in range(rp.shape[1]):
        list_rp.append(rp[:, i].sum())
        list_f.append(f[:, i].sum())


    list_rp = np.asarray(list_rp)
    list_f= np.asarray(list_f)
    print(list_f,list_rp)


    o=list_f
    e=list_rp
    arr = np.asarray([o, e])
    chi2, pvalue, dof, expected = stats.chi2_contingency(arr)
    chi_squared_stat = (np.power(o- e,2) / e).sum()
    pvalue = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                                 df=8)

    return pvalue

print(testchi2(val))