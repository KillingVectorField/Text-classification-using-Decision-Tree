import numpy as np
import tree_building
from tqdm import tqdm
import text_data
import time

def devide_data(data,k):
    '''devide data into k folds'''
    number_by_class=[len(x) for x in data]
    sample1_by_class=[x//k for x in number_by_class]
    np.random.seed(1)
    sample=[np.random.permutation(range(i)) for i in number_by_class]
    devided_data=[]
    for t in range(k-1):#前k-1组
        test=[[data[cls][j] for j in sample[cls][t*sample1_by_class[cls]:(t+1)*sample1_by_class[cls]]] for cls in range(len(data))]
        train=[[data[cls][j] for j in list(sample[cls][:t*sample1_by_class[cls]])+list(sample[cls][(t+1)*sample1_by_class[cls]:])] for cls in range(len(data))]
        devided_data.append((train,test))
    #第k组
    train=[[data[cls][j] for j in sample[cls][:(k-1)*sample1_by_class[cls]]] for cls in range(len(data))]
    test=[[data[cls][j] for j in sample[cls][(k-1)*sample1_by_class[cls]:]] for cls in range(len(data))]
    devided_data.append((train,test))
    return devided_data



def k_fold_CV(DATA,k,least_frequency,least_info_gained,stopping_proportion,IsForest=False):
    devided_data=devide_data(DATA,k)
    test_result=[]
    for data in tqdm(devided_data):
        all_vocab=tree_building.count_all(data[0])#全部词条（未删除低频词）
        selected_vocab=tree_building.select_variables(all_vocab,least_frequency)#忽略总共出现次数太小的词
        print('features个数：',len(selected_vocab))
        tree=tree_building.DecisionTree_Building(data[0],selected_vocab,least_info_gained,stopping_proportion)
        test_result.append(tree_building.Count_Predictions(tree,data[1]))
        print('\n',test_result[-1],'\n',tree_building.Correction_Rate(test_result[-1]))
    correct_rate=np.array([tree_building.Correction_Rate(count) for count in test_result])
    mean_correct_rate=np.mean(correct_rate)
    mixture_matrix=np.array([tree_building.Mixture_Matrix(count) for count in test_result])
    mean_mixture_matrix=np.mean(mixture_matrix,axis=0)
    test_result=[mean_correct_rate,mean_mixture_matrix.tolist()]
    text_data.save_json(test_result,str(k)+'-fold_CV_'+str(least_frequency)+r'_'+str(least_info_gained)+r'_'+str(stopping_proportion)+'.txt')
    return (mean_correct_rate,mean_mixture_matrix)
