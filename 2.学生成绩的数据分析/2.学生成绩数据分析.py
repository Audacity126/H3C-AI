#!/usr/bin/env python
# coding: utf-8

# In[2]:


#导入包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")  #忽略告警
plt.rcParams['axes.unicode_minus']=False #解决负号显示问题

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# In[3]:


df = pd.read_csv('exams.csv')
print(df.shape)
df_org=df.copy()
df.head()
#属性解读：race种族，associate's degree	副学士 


# In[4]:


#用总成绩代替读reading、写writing和数学math成绩，并输出预览前5行数据
df["总成绩"] = np.sum([df["reading score"], df["writing score"],df["math score"]], axis=0)
df.head()


# In[5]:


# 查看数据的特征、是否有缺失值、数据类型
df.info()
# 运行结果可知，数据集本身不存在缺失值。


# In[6]:


# 查看每一行是否存在重复值
df.duplicated()
# 运行结果可知，数据集每一行不存在重复值。


# In[7]:


#better way to check
duplicate_rows=df.duplicated()
print(df[duplicate_rows])


# In[8]:


#take a look at the unique values for each column
#for each column
for c in list (df.columns):
    #get a list of unique values
    n=df[c].unique()
    #in number of unique values is less than 30, print the values. Otherwise print the number of unique values.
    if len(n)<30:
        print(c)
        print(n)
    else:
        print (c+':'+ str(len(n))+'unique values')


# In[9]:


#数据分析
#(a)gender
#Series.plot() 方法可以用于绘制 Series 对象的可视化。这是 Pandas 中的一个方便的方法，它基于 Matplotlib 库提供了简单的绘图功能。
# 绘制以性别为变量的饼图
df['gender'].value_counts().plot.pie(autopct='%0.2f%%',colormap='GnBu')
# `colormap` 参数用于设置绘图时所使用的颜色映射（colormap）。颜色映射定义了数值到颜色的映射关系，使得图形中不同数值具有不同的颜色。
# `colormap` 参数通常是一个字符串，代表预定义的颜色映射，其中 `'GnBu'` 表示 "Green to Blue" 的颜色映射。这是一种渐变色映射，从绿色过渡到蓝色，用于显示数值变化的趋势。


# In[10]:


#(b)parental.level.of.education 
#绘制以父母教育水平为变量的饼图
df['parental level of education'].value_counts().plot.pie(autopct='%0.2f%%',colormap='Blues_r',figsize=(10,10))  


# In[11]:


# 父母学历对学生成绩的影响
#父母学历的条形图
sns.countplot(data=df,y="parental level of education")
# 显示图形
plt.show()
# 通过对父母学历条形图进行分析，some college数量最多，超过250，masters degree数量最少，数量大概在50-60之间。


# In[12]:


#(c)test.preparation.course 
#绘制以考前准备为变量的饼图
df['test preparation course'].value_counts().plot.pie(autopct='%0.2f%%',colormap='Pastel2')
# (a)表现了数据集中的男女占比大小，男性占比50.9%，女性占比49.1%，占比相差不大。(b)表现了父母的教育水平分布较为分散，存在六种种类，
# 占比最大的是some college为27.2%，占比最少的是master’s degree为6%。(c)表现了备考占比，
# 考前进行备考的占总体的37.2%，没有进行备考的占总体的62.8%，未进行备考的占大多数。


# In[13]:


# 绘制不同性别下的三种成绩的核密度图
#核密度图通常用于数据探索，帮助观察数据的主要趋势和模式。它也可以用于比较不同组或类别之间的数据分布。
# 设置多图形的组合,设置绘图风格,大小
plt.style.use('ggplot')
fig, axes = plt.subplots(3,1, figsize=(10,13))

#ax=axes[i]用于指定不同画布
# 绘制不同性别下的数学成绩核密度图
#绘图库的内部处理：有些绘图库在生成核密度图时可能采用了一些插值或平滑的技巧，以便更好地展示分布。这可能导致在实际数据范围之外的位置上看到曲线。
df['math score'][df.gender=='male'].plot(kind='kde',label='male',ax=axes[0],legend=True,linestyle='-')
df['math score'][df.gender=='female'].plot(kind='kde',label='female',ax=axes[0],legend=True,linestyle ='--')
axes[0].set_title('math score')
# 绘制不同性别的阅读成绩核密度图
df['reading score'][df.gender == 'male'].plot(kind='kde',label='male',ax = axes[1],legend = True,linestyle = '-')
df['reading score'][df.gender == 'female'].plot(kind='kde',label='female', ax = axes[1],legend = True,linestyle = '--')
axes[1].set_title('reading score')
# 绘制不同性别的写作成绩核密度图
df['writing score'][df.gender == 'male'].plot(kind='kde',label='male',ax = axes[2],legend = True,linestyle = '-')
df['writing score'][df.gender == 'female'].plot(kind='kde',label='female', ax = axes[2],legend = True,linestyle = '--')
axes[2].set_title('writing score')
# 显示图形
plt.show()
# 通过核密度图对学生的三个科目的成绩进行一个总体分布把握。
# 数学成绩的密度65左右达到峰值，其中男性的数学成绩整体上看高于女性，
# 阅读和写作成绩均在75左右达到峰值，女性在整体上对阅读和写作更具有优势。


# In[14]:


#特征工程
# 离散变量的重编码
#pd.Categorical().codes 适用于将一个列中的所有元素都转换为整数编码，而且通常用于处理分类数据列。
#与此类似的scikit-learn中LabelEncoder 更通用，可用于编码任何一维数组，不仅仅是 pandas 的 Series
for feature in df.columns:
    if df[feature].dtype == 'object':
        df[feature] = pd.Categorical(df[feature]).codes
        # 获取原属性和编码之间的对应关系        
        category_mapping = dict(zip(df_org[feature].unique(), df[feature].unique()))
        print(f"对应关系 ({feature}): {category_mapping}")
df.head()
#在Python中，f-string（格式化字符串字面值）是一种在字符串字面值中嵌入表达式的方法，使用花括号 {} 来包裹表达式。
# 这些表达式在运行时进行求值，它们的值会被格式化并插入到字符串中。


# In[15]:


# 属性间相关系数
#相关系数衡量了两个变量之间的线性关系的强度和方向。这是在数据分析和统计学中常用的一种工具，有助于理解变量之间的关联性。
cor = df.corr()
print(cor)

# 属性间相关系数热力图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12,9))
sns.heatmap(cor,annot=True,fmt=".2f",cmap='Blues')
plt.show()


# In[16]:


#数据建模与预测：线性回归、梯度下降、逻辑回归（分类）
#将数据集划分为测试集和训练集
x = df.drop(['总成绩',"math score","reading score","writing score"], axis=1) 
y = df['总成绩']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)# 划分数据集

print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('y_test shape: {}'.format(y_test.shape))
# 将数据集按照训练集和测试集4:1进行分配，方便后续数据建模及模型检验。
x_train.head(5)


# In[17]:


### 定义 线性回归模型 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()

# 使用训练数据进行参数估计
LR.fit(x_train, y_train)

#目标变量=截距+(回归系数1×特征1)+(回归系数 2×特征 2)+…+(回归系数n×特征n)
#截距对应于 LinearRegression.intercept_，而回归系数i 对应于 LinearRegression.coef[i]。
w = LR.coef_          #求系数
b = LR.intercept_     #求截距

print("w = ", LR.coef_," b= ", LR.intercept_)


# In[18]:


#计算均方误差MSE和R平方系数
y_predict=LR.predict(x_test)
# MSE 的值越小越好，表示模型的预测值与实际值的平方差异越小。
# R平方为1时，模型完美拟合；为0时，模型与简单平均相当；小于0时，说明模型拟合效果差于平均水平。
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_predict))
#R平方是一个统计量，表示模型解释的目标变量方差的比例
# R平方的取值范围在0到1之间，越接近1表示模型的解释能力越强，越接近0表示模型解释能力较差。
# R平方为1时，模型完美拟合；为0时，模型与简单平均相当；小于0时，说明模型拟合效果差于平均水平
print("Coefficient of determination: %.2f" % r2_score(y_test,y_predict))
# 如果关注预测值与实际值的差异，可以使用均方误差（MSE）。
# 如果更关注模型对目标变量方差的解释能力，可以使用R平方分数（R²）


# In[19]:


#手动计算MSE和R平方系数
y_predict=LR.predict(x_test)
r2_lasso=LR.score(x_test,y_test)
mse_lasso=np.average((y_predict-np.array(y_test))**2)
rmse=np.sqrt(mse_lasso)
print('MSE=',mse_lasso,'判定系数R2=',r2_lasso)


# In[20]:


# 使用Sklearn封装好的梯度下降回归SGDRegressor进行训练和预测
# 从sklearn.preprocessing导入数据标准化模块。
from sklearn.preprocessing import StandardScaler
# 特征值 标准化
x_train_tz=StandardScaler().fit_transform(x_train)
x_test_tz=StandardScaler().fit_transform(x_test)
x_train_tz


# In[21]:


# 从sklearn.linear_model导入SGDRegressor。
from sklearn.linear_model import SGDRegressor
SR=SGDRegressor()

# 使用训练数据进行参数估计
SR.fit(x_train_tz, y_train)

# 对测试数据进行预测。
y_test_pred=SR.predict(x_test_tz) 

# 性能评估
# The mean squared error   
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))

# The coefficient of determination: 1 is perfect prediction  #判定系数或R平方分数

print("Coefficient of determination: %.2f" % r2_score(y_test, y_test_pred))
# LinearRegression 适用于较小的数据集，且在计算资源充足的情况下，可以直接计算参数的闭式解。
# 而 SGDRegressor 更适用于大规模数据集，能够通过随机梯度下降迭代进行模型训练，具有更好的扩展性。
# 选择使用哪个模型取决于数据集的规模、计算资源以及是否需要正则化等因素。


# In[22]:


#可视化预测目标的真实值和预测值
plt.rcParams['font.sans-serif']=['simHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(facecolor= 'w',figsize=(10,6))
t = np.arange(len(x_test))
plt.plot(t,y_test,'r-',linewidth=2,label='y_test')
plt.plot(t,y_predict,'b-',linewidth=2,label='y_test_pred')
plt.legend(loc='upper left')
plt.show()


# In[23]:


#逻辑回归模型预测
# 若 Total score >= 250,则赋值为 1 （优秀）， 若 250> Total score >=200, 则赋值为 2（良好） ，若 Total score< 200,则赋值为3（一般）
def func(x):
    if x >=250:
        return "1"
    elif x >=200:
        return "2"
    else:
        return '3'
    
df["等级"] = df["总成绩"].apply(func)
df


# In[24]:


#将数据集划分为测试集和训练集
x = df.drop(['等级',"math score","reading score","writing score","总成绩"], axis=1) 
y = df['等级']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)# 划分数据集

print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('y_test shape: {}'.format(y_test.shape))
x_train.head(5)


# In[25]:


## 定义 逻辑回归模型 
from sklearn.linear_model import LogisticRegression
#'ovr'（默认值）：采用一对多（One-vs-Rest）的策略，将多类别问题拆分成多个二分类问题。
#并使用随机平均梯度下降（Stochastic Average Gradient Descent，'sag'）作为优化算法
model=LogisticRegression(multi_class='ovr',solver='sag')

# 在训练集上训练逻辑回归模型
model.fit(x_train,y_train)
#打印模型、模型系数、模型截距
print('模型','-'*20,'\n',model)
print('模型系数','-'*20,'\n',model.coef_)
print('模型截距','-'*20,'\n',model.intercept_)


# In[26]:


# 在测试集上利用训练好的模型进行预测
#如果关心预测的类别而不是概率，通常使用 predict() 方法。
y_predict=model.predict(x_test)
y_predict


# In[27]:


# 由于逻辑回归模型是概率预测模型，所以我们可以利用 predict_proba 函数预测其概率
#如果想了解模型对每个类别的置信程度，可以使用 predict_proba() 方法
test_predict_proba=model.predict_proba(x_test)
# 其中第一列代表预测为1类的概率，第二列代表预测为2类的概率，第三列代表预测为3类的概率
print('The test predict Probability of each class:\n',test_predict_proba)


# In[28]:


#打印真实值与预测值
#将预测结果y_predict转换为dataframe对象y_predict_df
y_predict_df=pd.DataFrame(y_predict,columns=['y_predict'],index=y_test.index)
#拼接测试集真实结果y_test与预测结果y_predict_df
y_test_predict_df=pd.concat([y_test,y_predict_df],axis=1)
print('     真实值与预测值','\n','-'*20,'\n',y_test_predict_df)


# In[29]:


#  利用accuracy（准确率）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('The accuracy of the Logistic Regression is:',accuracy_score(y_test,y_predict))


# In[30]:


# 利用混淆矩阵来评估模型效果
confusion_matrix_result=confusion_matrix(y_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)

# 用热力图对混淆矩阵进行可视化
plt.figure(figsize=(8,6))
# sns.heatmap(confusion_matrix_result,annot=True,cmap='Blues')
sns.heatmap(confusion_matrix_result,annot=True,cmap='Blues',xticklabels=[1, 2,3], yticklabels=[1, 2,3])

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
# 混淆矩阵可以直观看出模型的预测结果准确率，还可以依据混淆矩阵计算出查准率、查全率和F1值。


# In[31]:


# 利用查准率、查全率、F1 score 评估模型效果 
from sklearn.metrics import classification_report
target_names=['class 1','class 2','class 3']
classifyreport=classification_report(y_test,y_predict,target_names=target_names)
print('分类结果报告\n',classifyreport)
# Precision（精确率）：指在所有模型预测为正类别的样本中，实际为正类别的比例。计算公式为 Precision = TP / (TP + FP)。
# Recall（召回率）：指在所有实际正类别的样本中，被模型正确预测为正类别的比例。计算公式为 Recall = TP / (TP + FN)。
# F1-score（F1 分数）：综合考虑精确率和召回率的指标，是二者的调和平均数。计算公式为 F1 = 2 * (Precision * Recall) / (Precision + Recall)。
# 高 F1 分数：表示模型在保持高精确率和高召回率的同时，对该类别的分类性能较好。低 F1 分数：可能表示模型在精确率和召回率之间存在某种平衡问题，或者对该类别的分类性能较差。
# Support（支持度）：指每个类别在数据集中的样本数量。支持度可以帮助你了解每个类别的数据分布情况，即每个类别在数据集中有多少样本。
# Accuracy（准确率）：指模型正确分类的样本数量占总样本数量的比例。计算公式为 Accuracy = (TP + TN) / (TP + TN + FP + FN)。
# Macro avg（宏平均）：对所有类别的指标取平均，每个类别的权重相等。
# Weighted avg（加权平均）：对所有类别的指标进行加权平均，权重是每个类别的支持度（样本数量）。


# In[32]:


#对于一个新样本[gender	race/ethnicity	parental level of education	lunch	test preparation course]=[1,1,1,1,1]
# 采用已训练的模型model进行预测
x_new = np.array([[1,1,1,1,1]])
prediction =model.predict_proba(x_new)
pre=model.predict(x_new)
print("预测等级的概率：{}".format(prediction))
print("预测等级的类型: {}".format(pre))


# In[ ]:




