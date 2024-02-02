#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # 导入numpy库
import pandas as pd  # 导入pandas库
from sklearn.ensemble import GradientBoostingRegressor  # 集成方法回归库
from sklearn.model_selection import GridSearchCV  # 导入交叉检验库
import matplotlib.pyplot as plt  # 导入图形展示库
import matplotlib
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px  # 用于绘制漏斗图
# 解决中文乱码
plt.rcParams['font.sans-serif']=['simHei']
plt.rcParams['axes.unicode_minus'] = False


# In[2]:


raw_data = pd.read_table('products_sales.txt', delimiter=',')
raw_data.head()


# In[3]:


df = raw_data.copy()
df.rename(columns={'limit_infor': '是否限购', 'campaign_type': '促销类型', 'campaign_level': '促销级别','product_level': '产品等级', 
                   'resource_amount': '促销资源数', 'email_rate': '宣传投入', 'price': '单价', 'discount_rate': '折扣', 
                   'hour_resouces': '促销时长', 'campaign_fee': '促销费用', 'orders': '订单量'}, inplace = True)
df.head(5)


# In[4]:


df.info()


# In[5]:


# 在 "单价"字段中有 2 个缺失值。考虑到整个样本量比较少，因此做填充处理；而该列是连续型，选择填充为均值
sales_data = df.fillna(df['单价'].mean()) 


# In[6]:


#再次查看info，空值被补齐
sales_data.info()


# In[7]:


#使用 describe（）方法默认展示连续变量描述性统计结果，再使用 round(3) 只保留 3 位小数。
sales_data.describe().round(3)


# In[8]:


#利用数据可视化进行数据探索
price = sales_data['单价']
sns.distplot(price)
# 单价符合正态分布，这是一个比较好的性质。


# In[9]:


# “宣传投入”画出直方图
plt.hist(sales_data['宣传投入'], bins=20, edgecolor='k')
plt.xlabel('宣传投入')
plt.ylabel('频数')
plt.title('宣传投入分布直方图')
plt.show()


# In[10]:


#“宣传投入”和“折扣”的取值箱线图
# 在箱线图中，箱体为IQR=Q3-Q1
# 异常值被定义为小于Q1 - 1.5IQR或大于Q3 + 1.5IQR的值
sns.boxplot(data=sales_data[['宣传投入', '折扣']])
plt.title('宣传投入和折扣箱线图')
plt.show()
# 可见，“宣传投入”的中位数是0.5，“折扣”的中位数是0.8；“宣传投入”没有异常值，“折扣”存在多个异常小值。
# 结合实际，会出现打巨折的情况，所以该异常值不能简单的删除。


# In[11]:


category_counts = sales_data['促销类型'].value_counts()
# 绘制饼图
plt.figure(figsize=(6, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('促销类型饼图')
plt.axis('equal')  # 使图表呈现为正圆
plt.show()
#各个促销类型的占比差不多，在促销类型上为均衡样本


# In[12]:


# 对促销资源数画漏斗图
tmp=sales_data['促销资源数'].value_counts()
# 这里我们把离散数据当作类型变量画图，构建一个字典取对应数目及数量值
# 再使用px.funnel函数指明x轴变量和y轴变量画出，注意这里x和y的参数需要设置的与字典的key一致。
data=dict(数量=list(tmp),促销资源数=list(map(str,tmp.index)))
fig = px.funnel(data, x='数量', y='促销资源数')
# 显示图表
fig.show()
# 可见，大多数商品的促销资源数为中等水平3个，其次是7，6，4，最少的是1和9。毕竟大多数商品都需要促销资源，也几乎不可能拥有全线资源
# 漏斗图通常用于可视化一个过程中不同阶段的数据，展示从一个阶段到下一个阶段的数据递减或递增情况。漏斗图适用于以下一些场景：
# 销售漏斗： 可以用于表示销售过程中从潜在客户到实际销售的转化率，从而可视化销售团队的绩效。
# 转化漏斗： 用于表示在某个流程中，例如用户注册、激活、购买等步骤中的转化率，帮助识别可能的瓶颈和改进点。
# 营销效果： 可以用来展示在不同营销渠道中观察到的潜在客户到最终销售的转化情况，以优化营销策略。
# 招聘流程： 在人才招聘中，可以用漏斗图来展示从招聘流程的初筛、面试、录用等不同阶段的人才数量变化。
# 网站用户行为： 用于分析用户在网站上的行为，例如从浏览到注册、登录、购物车、结账等阶段的转化率。


# In[13]:


# 对于二元关系，我们可以使用散点图。这里我们关注的因变量是“订单量”，以“宣传投入”为例，做出它们的散点图。
plt.scatter(sales_data['宣传投入'], sales_data['订单量'])
plt.xlabel('宣传投入')
plt.ylabel('订单量')
plt.title('宣传投入和订单量关系散点图')
plt.show()
# 宣传投入和订单量关系有一定的正相关关系，也就是说，宣传投入越大，订单量越多，这一点也符合我们的认知


# In[14]:


#用热力图可视化展示连续变量之间的相关性。
correlation_matrix = sales_data.corr()
correlation_matrix


# In[15]:


# 绘制相关关系图
plt.figure(figsize=(8, 6))
# annot=True：这个参数表示是否在热力图中显示每个单元格的数值。设置为 True，则会在每个单元格中显示相关系数的数值。
# cmap='coolwarm'：这是用于设置热力图颜色映射的参数。'coolwarm' 是一个预定义的颜色映射，会使较小的值呈现冷色（蓝色），较大的值呈现暖色（红色）。
# center=0：这个参数设置热力图的中心值，用于确定颜色映射的中心。在这里，将中心值设置为 0，这意味着颜色映射的中心是白色，而负值偏向蓝色，正值偏向红色。
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('数值列之间的相关关系图')
plt.show()
# 通过相关性结果分析可知，不少变量之间存在强相关性，但本实验后续应用的是集成方法，因此没有对该特点做处理。


# In[16]:


# 将数据按照7:3分为训练集和测试集。
# 分割数据集X和y
num = int(0.7*sales_data.shape[0])   #sales_data.shape[0]获得样本数
X,y = sales_data.iloc[:, :-1],sales_data.iloc[:, -1]
X_train,X_test = X.iloc[:num,:],X.iloc[num:,:]
y_train,y_test = y.iloc[:num],y.iloc[num:]


# In[21]:


# GradientBoostingRegressor 是 scikit-learn 提供的用于回归问题的梯度提升算法的实现，它通过集成多个弱预测模型来构建一个更强大的预测模型，适用于许多回归任务。
# 梯度提升： 梯度提升是一种迭代的集成学习方法，每一步都在减小先前步骤残差（预测值与实际值之间的差异）的方向上构建新的学习器。这种方法通过反复训练弱模型来逐步改善整体模型的性能。
# 回归任务： GradientBoostingRegressor 用于解决回归问题，即预测连续数值的目标变量。与分类问题不同，回归任务的目标是预测实数值而不是离散标签。
# 弱学习器： 弱学习器是指在单独考虑时性能较差的模型，例如深度较浅的决策树。梯度提升通过组合多个弱学习器来形成强大的集成模型。
model_gbr = GradientBoostingRegressor()  # 建立GradientBoostingRegressor回归对象
parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],    #损失函数，可以是 'ls'（最小二乘回归，对应平方损失）、'lad'（最小绝对偏差回归）、'huber'（Huber损失，对异常值较为鲁棒）、'quantile'（分位数回归）等。这些损失函数衡量了模型预测与实际值之间的差异。
              'n_estimators': [10, 50, 100],                 #弱学习器的数量，即迭代次数。候选值包括 10、50、100。
              'learning_rate': [0.05, 0.1, 0.15],            #学习率，控制每个弱学习器的贡献程度。候选值包括 0.05、0.1、0.15。
              'max_depth': [2, 3, 4],                        #每个弱学习器的最大深度，防止模型过拟合。候选值包括 2、3、4。
              'min_samples_split': [2, 3, 5],                #每个内部节点分裂所需的最小样本数。候选值包括 2、3、5。
              'min_samples_leaf': [1, 2, 4]}                 # 每个叶子节点所需的最小样本数。候选值包括 1、2、4。
# 使用 GradientBoostingRegressor 模型，并通过网格搜索（Grid Search）来寻找最佳的超参数组合，以优化模型的性能。
# 在网格搜索（Grid Search）过程中，指定的候选值是要在模型训练的过程中尝试的不同参数值。
# 具体来说，对于每个参数，网格搜索会尝试不同的候选值，然后通过交叉验证来评估每组参数的性能。
# 这涉及多次训练和验证模型，每次都使用不同的参数组合。
model_gs = GridSearchCV(estimator=model_gbr,
                        param_grid=parameters, cv=3, n_jobs=1)  # 建立交叉检验模型对象
model_gs.fit(X_train, y_train)  # 训练交叉检验模型
# best_score_ 是使用交叉验证得到的 R^2 分数（判定系数）。R^2 分数是一种衡量模型拟合程度的指标，取值范围在 [0, 1] 之间，越接近1表示模型拟合得越好。
print('Best score is:', model_gs.best_score_)  # 获得交检验模型得出的最优得分
print('Best parameter is:', model_gs.best_params_)  # 获得交叉叉检验模型得出的最优参数


# In[18]:


# 知道最佳模型的参数后，我们可以选择手动赋值到模型，也可以更方便地使用 best_estimator_方法获得最佳模型对象。
model_best = model_gs.best_estimator_  # 获得交叉检验模型得出的最优模型对象
model_best


# In[19]:


from sklearn.metrics import r2_score
# 回归指标MSE评估
# MSE（均方误差，Mean Squared Error）是回归模型性能评价的一种指标，它衡量模型预测值与实际值之间的平均平方差。
# MSE 的数值越小越好，表示模型的预测值与实际值之间的差异越小，模型的性能越好。
pre_test = model_best.predict(X_test) #预测label
mse_score = mse(pre_test,y_test)
print(f"mse_score:{mse_score}")
r2_test = r2_score(y_test, pre_test)
print(f"r2_score:{r2_test}")


# In[20]:


# 模型拟合程度
plt.style.use("ggplot")  #将 Matplotlib 的绘图样式设置为 "ggplot" 风格。R语言中的ggplot 样式库强调数据可视化的美观性，提供了各种优雅的图形元素和颜色调色板，以增强可读性和吸引力
plt.figure(figsize=(10,7))  # 建立画布对象
plt.plot(np.arange(X_test.shape[0]), y_test, linestyle='-', color='k', label='true y')  # 画出原始变量的曲线
plt.plot(np.arange(X_test.shape[0]), pre_test, linestyle=':', color='m',
         label='predicted y')  # 画出预测变量曲线
plt.title('best model with mse of {}'.format(int(mse_score)))
plt.legend(loc=0)  # 设置图例位置
#测试集的预测值（粉色虚线）与真实值（黑色实线）的拟合程度比较高。


# In[ ]:




