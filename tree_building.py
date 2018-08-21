import os
import numpy as np
import text_data
#from tqdm import tqdm

#统计所有分类下的词汇出现次数
def count_all(data):
    vocab=dict()
    for cls in data:
        for item in cls:
            for word in item:
                if word not in vocab:
                    vocab[word]=1
                else:
                    vocab[word]+=1
    return sorted(vocab.items(), key=lambda d:d[1], reverse = True)#返回一个list

#统计单个分类下的词汇出现次数
def count_class(data):
    vocab=dict()
    for item in data:
        for word in item:
            if word not in vocab:
                vocab.update([(word,1)])
            else:
                vocab[word]+=1
    return dict(sorted(vocab.items(), key=lambda d:d[1], reverse = True))#返回一个字典

def select_variables(vocab,least_frequency):
    for i in range(len(vocab)):
        if vocab[i][1]<least_frequency:#忽略出现次数小于least_frequency的词，不将其作为一种feature
            vocab=vocab[:i]
            break
    return dict(vocab)

def integrated_count(selected_vocab,count_by_class):
    '''数出每个词再每一类中出现的次数'''
    for item in selected_vocab:
        distr=[]
        for i in range(len(count_by_class)):
            if item in count_by_class[i]:
                distr.append(count_by_class[i][item])
            else:distr.append(0)
        selected_vocab[item]=distr
    return selected_vocab

def cross_entropy(distr):
    '''输入某一（属性下的）分布，计算交叉熵'''
    Sum=0
    if not sum(distr)==0:#全都是0
        for x in distr:
            p=x/sum(distr)
            if not (x==0 or x==1):
                Sum-=p*np.log2(p)
    return Sum

def information_gained(init_distr, selected_distr):#属性为2值（有/无）
    '''训练样本集合按照一个属性划分后的信息增益'''
    init_entropy=cross_entropy(init_distr)
    selected_entropy=cross_entropy(selected_distr)
    unselected_distr=init_distr-selected_distr
    unselected_entropy=cross_entropy(unselected_distr)
    return init_entropy-selected_distr.sum()/init_distr.sum()*selected_entropy-unselected_distr.sum()/init_distr.sum()*unselected_entropy

def devide_by_property(data,str):
    '''根据有没有text中有无该'str'，把样本分为两堆'''
    data_with=[]
    data_without=[]
    for cls in data:
        cls_with=[]
        cls_without=[]
        for text in cls:
            if str in text:cls_with.append(text)
            else: cls_without.append(text)
        data_with.append(cls_with)
        data_without.append(cls_without)
    return (data_with,data_without)

def most_gained_property(init_distr,selected_vocab_distr):
    max_gain=-1
    selected_str=None
    for str in selected_vocab_distr:
        if information_gained(init_distr,np.array(selected_vocab_distr[str]))>max_gain:
            selected_str=str
            max_gain=information_gained(init_distr,np.array(selected_vocab_distr[str]))
    return (selected_str,max_gain)

def DecisionTree_Building(data,selected_vocab,least_info_gained,stopping_proportion):
    '''递归构建决策树
    data：当前训练样本集
    selected_vocab：要考虑的属性
    least_info_gained：最小信息收益
    stopping_proportion：最小停止比例
    '''
    number_by_class=np.array([len(cls) for cls in data])#每个分类下的样本数
    if number_by_class.max()/number_by_class.sum()>=stopping_proportion:#判停条件1：超过一定比例的样本都属于同一类
        return int(number_by_class.argmax())
    count_by_class=[count_class(data[cls]) for cls in range(len(data))]
    vocab_distr=integrated_count(selected_vocab,count_by_class)
    (selected,most_info_gained)=most_gained_property(number_by_class,vocab_distr)
    if most_info_gained<least_info_gained:#判停条件2：继续分类已经难以增加新的信息了
        return int(number_by_class.argmax())
    (data_with,data_without)=devide_by_property(data,selected)#将数据分为两堆
    del selected_vocab[selected]#不用再考虑这个属性了
    tree=[]
    tree.append(selected)
    tree.append(DecisionTree_Building(data_with,selected_vocab,least_info_gained,stopping_proportion))#添加左子树
    tree.append(DecisionTree_Building(data_without,selected_vocab,least_info_gained,stopping_proportion))#添加右子树
    if type(tree[1])==str and tree[1]==tree[2]:return tree[1]#后面的分叉导出相同的结果
    return tree

def predict(Decision_Tree,set,tell_class=False):
    '''用得到的树进行预测'''
    if type(Decision_Tree)==int:#分到类了
        if tell_class==False:
            return Decision_Tree
        else:
            return text_data.class_name[Decision_Tree]
    else:
        if Decision_Tree[0] in set:
            return predict(Decision_Tree[1],set,tell_class)
        else:
            return predict(Decision_Tree[2],set,tell_class)
        

'''
def Correct_Rate_of_Single_Class(tree,test,total_cls_num):
    #用同一类的测试集考察树的预测错误数和错误率
    count=np.array([0]* total_cls_num)
    for _ in test:
        count[predict(tree, _ ,False)]+=1
    return (count,count/len(test))
'''

def Count_Predictions(tree,test_data):
    '''用全体测试集考察树的预测错误数和错误率'''
    cls_num=len(test_data)
    count=np.zeros((cls_num,cls_num),dtype=np.float)
    for cls in range(len(test_data)):
        for _ in test_data[cls]:
            count[cls,predict(tree, _ ,tell_class=False)]+=1
    return count

def Mixture_Matrix(count):
    '''混合矩阵'''
    for row in count:
        row/=np.sum(row)
    return count

def Correction_Rate(count): return count.trace()/count.sum()

'''   
def RandomForest_Building(B,data,all_vocab,least_info_gained,stopping_proportion):
    #B是树的个数，all_vocab是列表，其中元素为('word',frequency)
    forest=[]
    for i in tqdm(range(B)):
        sample=np.random.choice(len(all_vocab),10*int(round(np.sqrt(len(all_vocab)))),replace=False)
        tmp_list=[]
        for j in sample:
            tmp_list.append(all_vocab[int(j)])
        selected_vocab=dict(tmp_list)
        forest.append(DecisionTree_Building(data,selected_vocab,least_info_gained,stopping_proportion))
    print('Random Forest Finished!')
    text_data.save_json(forest,r'rf_'+str(B)+r'_'+str(least_info_gained)+r'_'+str(stopping_proportion)+'.txt')
    return forest
'''

def count_nodes(tree):
    '''
    返回树的结点数和高度
    '''
    if type(tree)==int:#如果当前树只是一个叶子
        return (0,1)#分别为结点数和高度
    else:
        left_tree=count_nodes(tree[1])
        right_tree=count_nodes(tree[2])
        return (1+left_tree[0]+right_tree[0],1+max(left_tree[1],right_tree[1]))
