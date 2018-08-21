import os
import text_data
import numpy
import json
import tree_building
import cross_validation

least_frequency=10#总出现次数不超过least_frequency的词不考虑
least_info_gained=0.005
stopping_proportion=0.85
filename=r'tree_'+str(least_frequency)+r'_'+str(least_info_gained)+r'_'+str(stopping_proportion)+'.txt'

data=text_data.load_data()#读入各文件夹下的数据

number_by_class=[len(cls) for cls in data]#每个分类下的样本数
all_vocab=tree_building.count_all(data)#全部词条（未删除低频词）

selected_vocab=tree_building.select_variables(all_vocab,least_frequency)#忽略总共出现次数太小的词
print('features个数：',len(selected_vocab))
tree=tree_building.DecisionTree_Building(data,selected_vocab,least_info_gained,stopping_proportion)
print('Decision Tree Finished!')
scale=tree_building.count_nodes(tree)
print('结点数：',scale[0],"高度：",scale[1])
text_data.save_to_file('（可读）'+filename,str(tree))
text_data.save_json(tree,filename)
reload_tree=text_data.load_json(filename)
test={'柯震东'}
print(test,tree_building.predict(reload_tree,test,tell_class=True))
test_set=cross_validation.devide_data(data,10)[0][1]#产生一个测试集
Count_Predictions=tree_building.Count_Predictions(reload_tree,test_set)#直接在测试集上测试正确率
print(tree_building.Mixture_Matrix(Count_Predictions),tree_building.Correction_Rate(Count_Predictions))
text_data.save_json([len(selected_vocab),scale,tree_building.Correction_Rate(Count_Predictions),tree_building.Mixture_Matrix(Count_Predictions).tolist()],'(Total_Data) '+filename)
#记录的信息：可选属性个数，节点个数、树的高度，总正确率，总混合矩阵
print(cross_validation.k_fold_CV(data,10,least_frequency,least_info_gained,stopping_proportion))