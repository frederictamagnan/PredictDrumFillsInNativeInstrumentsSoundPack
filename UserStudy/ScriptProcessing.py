import pandas as pd
import re
import numpy as np
data= pd.read_csv('./userstudy.csv',dtype=str)
# data=data[data['Are you a musician ?']!='I\'m not a musician']
from scipy.stats import  ttest_rel
# def extractNumber(input):
#     # get a list of all numbers separated by
#     # lower case characters
#     # \d+ is a regular expression which means
#     # one or more digit
#     # output will be like ['100','564','365']
#     number =
#     return number[0]
def grade(data):
    for key in data.keys():

        data[key]=data[key].str.replace(' (very good fill)','', regex=False).str.replace(' (Very bad Fill)','',regex=False).str.replace(' (very bad fill)','',regex=False).str.replace(' ()','',regex=False).str.replace(' ','',regex=False)

    # print(data)
    grade=[0,0,0,0,0,0]
    grade_list=[[],[],[],[],[],[]]
    number=[0,0,0,0,0,0]
    list_value=[[],[],[],[],[],[]]


    id=[0,2,4,5,0,5,4,2,4,0,5,2,5,2,4,0,0,4,5,2]

    grade_column=0
    for i,key in enumerate(data.keys()):
        if "overall grade to these" in key:
            # print(key)
            # print(data[key])
            model_id=id[grade_column]

            sum_=pd.to_numeric(data[key]).sum(skipna=True)
            grade[model_id]+=sum_
            list_value[model_id]=list_value[model_id]+ pd.to_numeric(data[key]).tolist()

            count_=data.count()[i]
            # print(count_,"COUNT")
            number[model_id]+=(count_)
            grade_column+=1
            grade_list[model_id].append(sum_)

        if grade_column==3*4 :
            pass
    def remove_values_from_list(the_list, val):
       return [value for value in the_list if value != val]
    grade=remove_values_from_list(grade,0)
    number=remove_values_from_list(number,0)



    import numpy as np
    lol=np.array(grade)/np.array(number)
    print(lol)
    # print(grade_list)
    print(grade, number)
    return list_value,np.array(grade_list)
value,grade_list=grade(data)


# def most_frequent(List):
#     return max(set(List), key = List.count)


def most_frequent(List):
    unique_elements, counts_elements = np.unique(np.array(List), return_counts=True)
    return np.asarray((unique_elements, counts_elements/sum(counts_elements)))
mapping=[[0,2,4,5],[0,5,4,2],[4,0,5,2],[5,2,4,0],[0,4,5,2]]
total=[]
column_coherent=0
for i,key in enumerate(data.keys()):
    if "[Most coherent]" in key or "[most coherent]" in key or "[Most coherent ]" in key:
        clean=list((pd.to_numeric(data[key].str.replace('Sample','',regex=False),downcast='integer')))

        for elt in clean:
            try:
                # print(column_coherent,int(elt))
                total.append(mapping[column_coherent][int(elt)-1])
            except:
                pass
        column_coherent+=1

most=most_frequent(total)
print(most,"most coherent")



total=[]
column_coherent=0
for i,key in enumerate(data.keys()):
    if "[Less coherent]" in key or "[less coherent]" in key or "[Less coherent ]" in key:
        clean=list((pd.to_numeric(data[key].str.replace('Sample','',regex=False),downcast='integer')))

        for elt in clean:
            try:
                # print(column_coherent,int(elt))
                total.append(mapping[column_coherent][int(elt)-1])
            except:
                pass
        column_coherent+=1
        # print(column_coherent)
most=most_frequent(total)
print(most,"less coherent")


total=[]
column_coherent=0
for i,key in enumerate(data.keys()):
    if "[worst groove]" in key or "[Worst Groove]" in key or "[Worst groove ]" in key or "[Worst Groove ]" in key or "[Worst groove]" in key:
        clean=list((pd.to_numeric(data[key].str.replace('Sample','',regex=False),downcast='integer')))
        # print(column_coherent)
        for elt in clean:
            try:
                # print(column_coherent,int(elt))
                total.append(mapping[column_coherent][int(elt)-1])
            except:
                pass
        column_coherent+=1

most=most_frequent(total)
print(most,"worst groove")

total=[]
column_coherent=0
for i,key in enumerate(data.keys()):
    if "[best groove]" in key or "[Best Groove]" in key or "[Best groove]" in key or "[Best Groove ]" in key or "[Best groove ]" in key:
        clean=list((pd.to_numeric(data[key].str.replace('Sample','',regex=False),downcast='integer')))

        for elt in clean:
            try:
                # print(column_coherent,int(elt))
                total.append(mapping[column_coherent][int(elt)-1])
            except:
                pass
        column_coherent+=1
        # print(column_coherent)
most=most_frequent(total)
# print(most,"Best groove")
print(len(value[2]),len(value[4]))
print(np.isnan(value[4]).sum())
from scipy.stats import wilcoxon
stat,pvalue=ttest_rel(value[2],value[4],nan_policy='omit')

print(pvalue)
a=value[2]
b=value[4]
a=np.array(a)
b=np.array(b)
a=a[~(np.isnan(a))]
b=b[~(np.isnan(b))]
t,pvalue=wilcoxon(a,b)
print(pvalue)
print(b[:5])

print(grade_list)