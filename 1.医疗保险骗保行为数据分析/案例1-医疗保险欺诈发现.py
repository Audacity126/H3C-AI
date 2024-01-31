#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
policy_holder = pd.read_csv('Policy_Holder.csv',encoding = 'GBK')  # 投保人信息表
provider = pd.read_csv('Provider.csv',encoding = 'GBK')  # 医疗机构信息表
claim = pd.read_csv('Claim.csv',encoding = 'GBK')  # 索赔信息表


# In[2]:


print(policy_holder.head)
print(policy_holder.shape)
print(provider.head)
print(provider.shape)
print(claim.head)
print(claim.shape)


# In[3]:


# 描述性统计分析，返回缺失值个数、最大值、最小值
print("policy_holder.describe:\n",policy_holder.describe())
explore_policy_holder = policy_holder.describe(percentiles=[], include='all').T
print("explore_policy_holder:\n",explore_policy_holder)
explore_provider = provider.describe(percentiles=[], include='all').T  
explore_claim = claim.describe(percentiles=[], include = 'all').T 
# percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）
explore_policy_holder['null'] = policy_holder.isnull().sum()# 计算空缺值
explore_policy_holder = explore_policy_holder[['null', 'max', 'min']]
explore_provider['null'] = provider.isnull().sum()
explore_provider = explore_provider[['null', 'max', 'min']]
explore_claim['null'] = claim.isnull().sum()
explore_claim = explore_claim[['null', 'max', 'min']]


# In[4]:


explore_policy_holder


# In[5]:


explore_provider


# In[6]:


explore_claim


# In[7]:


#数据清洗
# 将非字符型投保人编号、医疗机构编号、索赔编号的特征转为字符型
policy_holder[['Policy_HolderID']] = policy_holder[['Policy_HolderID']].astype(str)
provider[['ProviderID']] = provider[['ProviderID']].astype(str)
claim[['索赔编号']] = claim[['索赔编号']].astype(str)
claim[['医疗机构编号']] = claim[['医疗机构编号']].astype(str)
claim[['投保人编号']] = claim[['投保人编号']].astype(str)

# 更改表的列名为中文，增强可读性
policy_holder.columns = ['投保人编号', '保险条款', '治疗措施编码', '年龄', '性别']
provider.columns = ['医疗机构编号', '医疗机构大类', '医疗机构细类', '位置编码']

# 住院开始时间、住院结束时间的特征转为时间类型转为时间类型
claim['住院开始时间'] = pd.to_datetime(claim['住院开始时间'], format = '%Y-%m-%d')
claim['住院结束时间'] = pd.to_datetime(claim['住院结束时间'], format = '%Y-%m-%d')


# In[8]:


print(policy_holder.head)
print(provider.head)


# In[9]:


# 提取保险条款数据
# policy_holder.describe()
ProgramCode = pd.value_counts(policy_holder['保险条款'])
#ProgramCode为series, 行标签为保险类别，数值对应为该保险条目数
print(ProgramCode,type(ProgramCode))
plt.figure(figsize=(5, 5))  # 设置画布大小

# 通过绘制饼图查看投保人所投保险条款的比例
plt.pie(ProgramCode,
        labels=ProgramCode.index, autopct='%1.2f%%')  
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('保险条款种类的比率饼图')
plt.legend()  # 图例
plt.show()


# In[10]:


# 提取并统计治疗措施编码数据
MEDcode = pd.value_counts(policy_holder['治疗措施编码'])
plt.figure(figsize = (8, 5))
plt.bar(MEDcode.index, MEDcode, width = 0.4,color='skyblue')
plt.xlabel('治疗措施编码')
plt.ylabel('频数')
plt.title('治疗措施编码种类的条形图')
plt.show()
#由条形图可以看出，投保人所对应的治疗措施共有 5 种，其中编码为 RegularMedicare
# 的治疗措施所占比例最高，说明投保人所接受的治疗措施中，常规的药物治疗最为常见。其
# 他的治疗措施所占比例相对低，提供给针对相对应治疗的投保病人。


# In[11]:


# 提取投老年保障险的投保人数据
#利用dataframe的loc方法进行标签筛选‘老年保障险’
old_distribute = policy_holder.loc[
        policy_holder['保险条款']=='老年保障险', :]
#按年龄统计样本数
old_distribute_age = pd.value_counts(old_distribute['年龄'])
plt.figure(figsize=(12, 8))
plt.subplot(121)  # 子图
plt.bar(old_distribute_age.index, old_distribute_age)
plt.title('老年保障险分布情况')
plt.xlabel('年龄')
plt.ylabel('频数')

# 提取投伤残险的投保人数据
insurance_disability = policy_holder.loc[
        policy_holder['保险条款']=='伤残险',:]
insurance_disability_age = pd.value_counts(insurance_disability['年龄'])
plt.subplot(122)  # 子图
plt.bar(insurance_disability_age.index, insurance_disability_age)
plt.title('伤残险分布情况')
plt.xlabel('年龄')
plt.ylabel('频数')
plt.show()
plt.close
# 由条形图可以看出，老年保障险和伤残险的投保人中，年龄分布有较大差异。其中伤残
# 险的投保人一般无年龄特别小者，不小于 16 岁；而老年保障险的投保人年龄则有出现比较小的情况，有小至 3 岁的投保人。
# 这个现象说明发生意外事故等伤残情况，相较之下容易发生在少年阶段及其以上的人群；
# 而老年保障险有年龄小的投保人，表明该保险较受有孩子家庭的欢迎


# In[12]:


# 提取并统计医疗机构大类数据
provider_type = pd.value_counts(provider['医疗机构大类'])
plt.figure(figsize=(10, 6))
plt.bar(provider_type.index, provider_type)  # 绘制医疗机构大类类别分布图
plt.xticks(rotation=90)
plt.title('医疗机构大类分布图')
plt.xlabel('医疗机构大类')
plt.ylabel('频数')
plt.show()
# 从图中可以看出内科的各种疾病的出险频率更高。


# In[13]:


# 根据保险行业的知识，投保人的出险模式和索赔模式在一段时间内不应该有较大变化，在上半年和下半年出险模式
# 和索赔模式变化较大的，疑似骗保。医疗机构的索赔模式在一段时间内也应该一致，在上半年和下半年变化较大的，疑似骗保。
import pandas as pd
#筛选核心属性'医疗机构编号', '投保人编号', '住院开始时间','保费覆盖额', '账单金额', '支付金额'
claim_money = claim.loc[:, ['医疗机构编号', '投保人编号', '住院开始时间',
                            '保费覆盖额', '账单金额', '支付金额']]

# 提取住院时间月份
claim_money['住院开始时间'] = claim_money['住院开始时间'].dt.month 
claim_money['住院开始时间'] = claim_money['住院开始时间'].astype(int)  #转换为int方便比较

# 按住院开始时间划分上半年和下半年数据
claim_money['所属时间段'] = 0
claim_money.loc[claim_money['住院开始时间'] <= 6, '所属时间段'] = '1H'
claim_money.loc[claim_money['住院开始时间'] > 6, '所属时间段'] = '2H'


# In[14]:


claim_money.head(5)


# In[15]:


#按['投保人编号','所属时间段']group by,筛选得到行索引
print(claim_money.groupby(['投保人编号','所属时间段']).groups)


# In[16]:


#针对['医疗机构编号']做计数
print(claim_money.groupby(['投保人编号','所属时间段'])['医疗机构编号'].count())


# In[17]:


# 根据投保人编号统计上下半年的索赔支付笔数总数 
claim_paycount = claim_money.groupby(['投保人编号','所属时间段'])['医疗机构编号'].count()
claim_paycount = claim_paycount.reset_index()
print("after reset index:\n",claim_paycount)#原列索引【医疗机构编号】不再适用


# In[18]:


#重命名列索引
claim_paycount.columns = ['投保人编号', '所属时间段', '半年支付笔数'] 
claim_paycount


# In[19]:


# 根据投保人编号统计上下半年保费覆盖额,账单金额,支付金额总额
claim_moneysum = claim_money.groupby(['投保人编号','所属时间段'])['保费覆盖额', '账单金额', '支付金额'].sum()
claim_moneysum = claim_moneysum.reset_index()
#重命名列索引
claim_moneysum.columns = ['投保人编号', '所属时间段',
                         '半年保费覆盖额','半年账单金额'
                         ,'半年支付金额'] 
#基于键['投保人编号','所属时间段']，合并claim_moneysum，claim_paycount
claim_part = pd.merge(claim_moneysum, claim_paycount,on=['投保人编号','所属时间段'])

claim_part


# In[20]:


# 选取投保人信息表中的年龄（Age）、性别（Sex 取值范围为女和男）、治疗措施编码、保险条款属性的特征，
# 除了年龄之外的特征采用虚拟变量法（One-hot,独热码），即按值进行展开：
#采用虚拟变量的原因：
#一些模型、特别是线性模型容易被数值大小关系所影响，为了避免模型在类别之间引入不必要的偏好，使用虚拟变量可以确保每个类别有自己独立的二进制表示
#某些某型要求输入为数值型，而不能是分类型数据，例如：逻辑回归

policy_holder_t = policy_holder
#原表policy_holder纵向double拼接
policy_holder_t = pd.concat([policy_holder_t, policy_holder], axis = 0)
policy_holder_t['所属时间段'] = 0   #增加一列'所属时间段'
#设置index索引
policy_holder_t.index = range(len(policy_holder_t))

# 确保每个投保人有两条记录
policy_holder_t.loc[0:400, '所属时间段'] = '1H'     #前400条标识为上半年'1H'
policy_holder_t.loc[400:800, '所属时间段'] = '2H'   #后400条标识为下半年'2H'
#可以采用on='投保人编号',如果没用指定On，默认采用共同列做连接键
claim_policy = pd.merge(claim_part, policy_holder_t,  how='outer')
print("claim_part:\n",claim_part.head(5))
print("policy_holder_t:\n",policy_holder_t.head(5))
print("after merge:\n",claim_policy.head(5))


# In[21]:


print(claim_policy.isnull().sum())
#以0填充空值
claim_policy = claim_policy.fillna(0)


# In[22]:


# 构建虚拟变量
model3_data1 = pd.concat([claim_policy.loc[:,['半年保费覆盖额', '半年账单金额', '半年支付金额','半年支付笔数','年龄']],
                          pd.get_dummies(claim_policy['性别']),
                          pd.get_dummies(claim_policy['治疗措施编码']),
                          pd.get_dummies(claim_policy['保险条款'])],axis = 1)
model3_data = model3_data1.copy()
model3_data


# In[23]:


# 医疗机构特征变换过程与投保人特征变换过程操作类似，同样数据都对时间进行划分，具体特征变换过程如下3点。
# 先根据索赔信息表中投保人的住院开始时间特征划分为上半年（1H）和下半年（2H）两个部分。
# 接着按医疗机构编号和所属时间段进行分组，统计投保人数和处理过程数量；选取索赔订单中保费覆盖额、账单金额、支付金额的特征，分布按上、下半年时间进行统计，分别得到半年保费覆盖额、半年账单金额、半年支付金额、半年支付笔数。再选取医疗机构信息表中医疗机构大类（ProviderType）、医疗机构细（ProviderSpecialty）、位置编码（Location）等3个特征。
# 发现医疗机构欺诈的特征变换
from sklearn.preprocessing import LabelEncoder
provider_t = provider.copy()
provider_t = pd.concat([provider_t, provider], axis=0 )
provider_t['所属时间段'] = 0
provider_t.index = range(len(provider_t))

# 确保每个投保人有两条记录
provider_t.loc[:500, '所属时间段'] = '1H'
provider_t.loc[500:1000, '所属时间段'] = '2H'

# 根据投保人编号统计上下半年保费覆盖额,账单金额,支付金额总额
money_sum = claim_money.groupby(['医疗机构编号', '所属时间段'])['保费覆盖额', '账单金额', '支付金额'].sum()
money_sum = money_sum.reset_index()
money_sum.columns = ['医疗机构编号', '所属时间段', '半年保费覆盖额','半年账单金额', '半年支付金额']

# 汇总处理过程数量
# 根据投保人编号统计上下半年保费覆盖额,账单金额,支付金额总额
claim_money['处理过程代码'] = claim['处理过程代码']
claim_money['处理过程代码'] = claim_money['处理过程代码'].astype(str)
count1 = claim_money.groupby(['医疗机构编号', '所属时间段'])['处理过程代码'].count()
deal_count = count1.reset_index()
deal_count.columns = ['医疗机构编号', '所属时间段', '处理过程数量']

# 根据投保人编号统计上下半年保费覆盖额,账单金额,支付金额总额
claim_money['投保人编号'] = claim_money['投保人编号'].astype(str)
count2 = claim_money.groupby(['医疗机构编号', '所属时间段'])['投保人编号'].count()
people_count = count2.reset_index()
people_count.columns = ['医疗机构编号', '所属时间段', '投保人数量']
people_deal = pd.merge(people_count, deal_count,on = ['医疗机构编号', '所属时间段'])
pd_money = pd.merge(people_deal, money_sum, on=['医疗机构编号', '所属时间段'])
claim_provider = pd.merge(pd_money, provider_t, how='outer')
claim_provider = claim_provider.fillna(0)
claim_provider_original = claim_provider.copy()


# In[24]:


#原始claim_provider['医疗机构大类']属性数值为分类属性
claim_provider['医疗机构大类'] .values


# In[25]:


# 由于医疗机构大类和医疗机构细类取值都为字符串类型，需要把字符串类型转为数值型，使用sklearn.preprocessing库下LabelEncoder函数将非数字型标签值标准化。
#LabelEncoder()用于生成单列整数标签（适用于决策树、支撑向量机）
#区别于之前的get_dummies()用于生成二进制列的矩阵（适用于线性回归、神经网络，避免引入虚假类别顺序）
class_le = LabelEncoder() # 将非数字型标签值标准化
claim_provider['医疗机构大类']  = class_le.fit_transform(claim_provider['医疗机构大类'] .values)+1
claim_provider['医疗机构细类']  = class_le.fit_transform(claim_provider['医疗机构细类'] .values)+1
print(claim_provider['医疗机构大类'])
claim_provider


# In[26]:


model3_data1.head(5)


# In[27]:


# 对投保人特征进行 K-Means 聚类。
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#sklearn.preprocessing.StandardScaler用于对特征进行标准化，标准化后的数据有利于提高模型收敛速度，避免某些特征对模型训练产生过大的影响
scaler = StandardScaler()
#fit用于计算模型的均值和方差
scaler.fit(model3_data1)
#transform利用前面计算得到的均值、方差，对数据进行标准化
model3_data = scaler.transform(model3_data1)
#也可合并使用方法fit_transform()一次性完成计算和转换

#转换后数据类型为numpy.ndarray，不再可使用head()方法，改为切片查看前5行
model3_data[:5]


# In[28]:


#数据准备完毕，进行kmeans模型训练
kmeans = KMeans(n_clusters = 5,random_state = 0) # K-Means聚类
kmeans.fit(model3_data)
#label_pred为kmeans训练得到的分类标签序列
label_pred = kmeans.labels_
#model3_centers1为聚类中心
model3_centers1 = kmeans.cluster_centers_
model3_centers = model3_centers1.copy()
print(model3_centers1)
print('model3_centers1.shape[1]:\n',model3_centers1.shape[1])


# In[29]:


#shape()方法返回矩阵规模，即：（行，列），shape[1]则返回该矩阵列数，即特征数量
#针对每个特征进行逆标准化操作，目的是讲聚类中心的值还原回原始数据的尺度,便于更好地理解聚类中心在原始数据中的含义
#逆标准化公式为（X*(max-min)）+min
for i in range(model3_centers1.shape[1]):
    model3_centers[:,i] = model3_centers1[:,i]*(max(model3_data1.iloc[:,i]) - 
                  min(model3_data1.iloc[:,i])) + min(model3_data1.iloc[:,i])
model3_centers


# In[30]:


claim_policy['cluster'] = label_pred  # 将Kmeans模型得到的分类标签序列添加为claim_policy新属性['cluster']
claim_policy


# In[31]:


claim_provider.head(5)


# In[32]:


# 对医疗机构特征进行 K-Means 聚类。
scaler = StandardScaler()
#模型输入属性排除‘医疗机构编号’、'所属时间段'属性，对聚类无意义的属性
#注意df.loc和iloc的用法区别
#scaler数据标准化
#fit用于计算模型的均值和方差，#transform利用前面计算得到的均值、方差，对数据进行标准化，#也可合并使用方法fit_transform()一次性完成计算和转换
scaler.fit(claim_provider.iloc[:,2:10])
claim_provider.iloc[:,2:10] = scaler.transform(claim_provider.iloc[:,2:10])

#构建、训练kmeans模型
model4_km  = KMeans(n_clusters = 5,random_state = 0)  # K-Means聚类
model4_km.fit(claim_provider.iloc[:,2:10])  # 预测
#保存聚类标签和聚类中心
model4_label_pred = model4_km.labels_
model4_centers1 = model4_km.cluster_centers_
#为数据集claim_provider添加聚类分群['cluster']
claim_provider['cluster'] = model4_label_pred
claim_provider


# In[33]:


#不同聚类的支付金额和支付笔数均值
claim_cluster = claim_policy.groupby('cluster')['半年支付金额','半年支付笔数'].mean()
claim_cluster


# In[34]:


#添加行索引index
claim_cluster = claim_cluster.reset_index()
#重命名列索引
claim_cluster.columns = ['类群编号','支付金额_mean','支付笔数_mean']
#统计并添加属性'投保人数量'
claim_cluster['投保人数量'] = pd.value_counts(claim_policy['cluster'])
claim_cluster


# In[35]:


#针对可疑投保人的筛选
claim_policy.head(5)


# In[36]:


#上半年cluster分群结果model3_result1,标签检索df.loc
model3_result1 = claim_policy.loc[claim_policy['所属时间段'] == "1H", :]
#下半年cluster分群结果model3_result2
model3_result2 = claim_policy.loc[claim_policy['所属时间段'] == "2H", :]
#重命名下半年数据集列属性
model3_result2.columns = ['投保人编号', '所属时间段_2','保费覆盖额_halfyear_2',
                          '账单金额_halfyear_2', '支付金额_halfyear_2','支付笔数_halfyear_2'
                          , '保险条款_2', '治疗措施编码_2', '年龄_2', '性别_2', 'cluster_2']

# 上半年、下半年数据集按['投保人编号']进行合并
model3_result = pd.merge(model3_result1, model3_result2, on=['投保人编号'])  
#pandas.crosstab探索不同聚类簇的变化关系
model3_migrate= pd.crosstab(model3_result ['cluster'], model3_result['cluster_2'])
model3_migrate


# In[37]:


pd.set_option('display.max_columns', None) #列属性不省略
model3_result.head(5)


# In[39]:


#筛选出上半年与下半年聚类分群不同的样本
model3_ysqz = model3_result.loc[model3_result.loc[:, 'cluster'] !=  model3_result.loc[:, 'cluster_2'],:]
model3_ysqz = model3_ysqz.loc[:,['投保人编号', 'cluster', 'cluster_2']]
#tmp目录需已存在
model3_ysqz.to_csv('tmp/policy_holder_ysqz.csv', index=False)
model3_ysqz.head()


# In[40]:


#针对可疑医疗机构的筛选
claim_provider_original.head()


# In[41]:


import pandas as pd
#利用前述kmeans聚类的分群结果，添加属性列['类群编号'] 
claim_provider_original['类群编号'] = model4_label_pred
#按类群统计样本数
collect_compare = pd.value_counts(claim_provider_original['类群编号'])
collect_compare


# In[42]:


collect_compare.index


# In[43]:


#series.reset_index()将原series索引作为列，并重置新的索引
collect_compare = collect_compare.reset_index()
collect_compare.columns = ['类群编号', '医疗机构数量']
collect_compare = collect_compare.sort_values(by='类群编号')
collect_compare


# In[44]:


collect_compare.index = range(len(collect_compare))
collect_compare


# In[45]:


collect_compare['平均投保人数量']  = claim_provider_original.groupby(['类群编号'])['投保人数量'].mean()
print(collect_compare)


# In[46]:


# 计算不同类群间账单金额的平均值
collect_compare['平均账单金额'] = claim_provider_original.groupby(['类群编号'])['半年账单金额'].mean()
print(collect_compare)


# In[47]:


# 划分上半年数据
model4_result1 = claim_provider_original.loc[claim_provider_original['所属时间段'] == '1H',: ]
model4_colunms = ['医疗机构编号', '投保人数量','处理过程数量',
                  '半年保费覆盖额','半年账单金额','半年支付金额', '类群编号']
model4_result1 = model4_result1.loc[:, model4_colunms] 
#上半年属性重命名
model4_result1.columns = ['医疗机构编号', '投保人数量1', '处理过程数量1',
                          '保费覆盖额1', '账单金额1', '支付金额1', '类群编号1']
model4_result1


# In[48]:


# 划分下半年数据
model4_result2 = claim_provider_original.loc[
        claim_provider_original['所属时间段'] == '2H',: ]
model4_result2  = model4_result2.loc[:, model4_colunms]   
#上半年属性重命名
model4_result2.columns = ['医疗机构编号', '投保人数量2', '处理过程数量2',
                          '保费覆盖额2', '账单金额2', '支付金额2', '类群编号2']
model4_result2


# In[49]:


#合并上下半年的数据
model4_result = pd.merge(model4_result1,model4_result2, on=['医疗机构编号']) 
#筛选可疑医疗机构：筛选出上半年与下半年分群不同的医疗机构
model4_ysqz = model4_result.loc[model4_result.loc[:, '类群编号1'] != model4_result.loc[:, '类群编号2'],:]
print("model4_ysqz：",model4_ysqz)


# In[50]:


#查看分群的迁移情况
model4_migrate = pd.crosstab(model4_result ['类群编号1'],model4_result ['类群编号2']) #查看类群迁移情况
print(model4_migrate)


# In[51]:


# 输出疑似欺诈的情况
model4_ysqz= model4_ysqz.loc[:,['医疗机构编号', '类群编号1', '类群编号2']]
model4_ysqz.to_csv('tmp/Provider_ysqz.csv', index=False)
model4_ysqz.head()


# In[52]:


# 由于原始数据本身不存在真实分类结果，发现投保人疑似欺诈和发现医疗机构疑似欺诈
# 都是基于 K-Means 聚类算法，故分类效果性能度量采用轮廓系数评价法和 Calinski-Harabasz 指标评价法对聚类模型进行评价
# 计算对于投保人聚类两种指标的值：
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore') 
# 轮廓得分(Silhouete Score)是评价聚类算法性能的一种指标，它结合了簇内平均距离和簇间平均距离，值越大表示聚类效果越好.
#如果轮廓系数接近1，说明数据点与其分配的簇相似，而与其他簇差异较大。
silhouettteScore3 = []
# CH指标通过计算类中各点与类中心的距离平方和来度量类内的紧密度，通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度，
# CH指标由分离度与紧密度的比值得到。从而，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。
# - `calinski_harabasz_score` 更侧重于全局聚类结构，关注簇的紧密度和分离度。
# - `silhouette_score` 更关注单个数据点与其分配的簇之间的相似度，更侧重于簇内的相似性。
calinskiharabazScore3 = []
#测试分群数2-12个分群，以比较不同聚类数量的模型表现
for i in range(2, 12):
    kmeans_num31 = KMeans(n_clusters = i, random_state = 0).fit(model3_data)  # 针对不同分群i,构建并训练模型
    score = silhouette_score(model3_data, kmeans_num31.labels_)  # 轮廓系数
    silhouettteScore3.append(score)
    print('数据聚%d类silhouette_score指数为：%f'%(i,score))
    score1 = calinski_harabasz_score(model3_data, kmeans_num31.labels_)
    calinskiharabazScore3.append(score1)
    print('数据聚%d类calinski_harabaz指数为：%f'%(i, score1))


# In[53]:


# 绘制折线图   
plt.figure(figsize=(8, 5))
plt.rcParams['font.sans-serif'] = 'SimHei'  # 显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1, 2, 1)  # 子图
plt.plot(range(2, 12),silhouettteScore3, 'ko-', linewidth=1.5)
plt.title('轮廓系数评价法')
plt.xlabel('聚类数目')
plt.ylabel('分数')
plt.subplot(1,2,2)  
plt.plot(range(2, 12), calinskiharabazScore3, 'o-', linewidth=1.5)
plt.title('Calinski-Harabasz指标评价法')
plt.xlabel('聚类数目')
plt.ylabel('分数')
plt.savefig('tmp/policy_holder_ysqz.jpg') 
plt.show()
plt.close
#如图所示，聚类数从2到3时和从9到10、10到11时模型性能相对下降较明显。
#Silhouette 分数在聚类数为2到11时都表现得相对较好，逐渐增加。这可能表示不同聚类数下，数据点之间的相似性和簇的分离度都在一定程度上得到了保持。
# 但从数据看，聚类数为8时，Silhouette 分数达到了一个较高的值，而聚类数进一步增加到9和10时，Silhouette 分数也有轻微提高。
# 考虑到轮廓系数的含义，通常更高的值表示更好的聚类效果。因此，可以初步认为在聚类数为8时，模型性能较好。
# 增加聚类数可能导致轻微的改善，但也可能增加模型的复杂性和理解的困难度
#聚类数的选择还应该考虑：
# 1. **计算复杂度：** 增加聚类数会导致算法需要计算更多的簇中心和分配数据点到相应簇的任务。一些聚类算法在处理大量簇时可能会增加计算的复杂性。
# 2. **可解释性：** 随着簇的增加，理解和解释每个簇的特性变得更加复杂。这可能需要更多的工作来解释模型中每个簇所代表的数据模式。
# 3. **业务解释：** 在业务层面，理解和解释更多的簇可能需要更多的时间和努力，因为每个簇的含义和业务影响可能需要更深入的分析。


# In[54]:


#医疗机构聚类性能评价
silhouettteScore4 = []  # 轮廓系数
calinskiharabazScore4 = []  # Calinski-Harabasz
for i in range(2, 12):
    kmeans_num41 = KMeans(n_clusters = i, random_state=0).fit(
            claim_provider.iloc[:, 2:10])  # 构建并训练模型
    score = silhouette_score(
            claim_provider.iloc[:, 2:10], kmeans_num41.labels_)
    silhouettteScore4.append(score)
    print('数据聚%d类silhouette_score指数为：%f'%(i,score))
    score1 = calinski_harabasz_score(claim_provider.iloc[:, 2:10],
                                    kmeans_num41.labels_)
    calinskiharabazScore4.append(score1)
    print('数据聚%d类calinski_harabaz指数为：%f'%(i,score1))


# In[55]:


# 绘制折线图
plt.figure(figsize=(8, 5))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1,2,1)  # 子图
plt.plot(range(2,12),silhouettteScore4, 'ko-', linewidth=1.5)
plt.title('轮廓系数评价法')
plt.xlabel('聚类数目')
plt.ylabel('分数')
plt.subplot(1, 2, 2)  
plt.plot(range(2, 12), calinskiharabazScore4, 'o-', linewidth=1.5)
plt.title('Calinski-Harabasz指标评价法')
plt.xlabel('聚类数目')
plt.ylabel('分数')
plt.savefig('tmp/Provider_ysqz_evaluate.jpg')
plt.show()
plt.close
# 1. **聚类数为2时：Silhouette 分数和 Calinski-Harabasz 分数都相对较高，这可能表示在2个聚类中，数据点内部相似度高，而簇之间的分离度也明显。
# 2. **聚类数为3时：Silhouette 分数和 Calinski-Harabasz 分数仍然较高，显示了较好的聚类效果。相对于2个聚类，增加到3个聚类，模型性能似乎有提升。
# 3. **聚类数超过5时：随着聚类数的增加，Silhouette 和 Calinski-Harabasz 分数逐渐下降。这可能表示在超过5个聚类时，模型性能有所减弱。
# 1. **Calinski-Harabasz 指数在聚类数为5时，Calinski-Harabasz 分数较高，表明模型整体上在簇内部更紧凑，簇间分离性较好。
# 2. **Silhouette 分数在聚类数为5时也相对较高，显示了较好的聚类效果。虽然比聚类数为3时稍低，但仍然维持在一个相对较好的水平。
# 3. **业务解释性： 聚类数为5时可能带有更多的细节，更复杂，因此在解释性方面可能相对较高。聚类数为3时模型更简单，可能更容易解释。
# 综合考虑这些因素，如果模型的复杂性可以被接受，并且业务需求需要更详细的聚类，那么聚类数为5可能是一个很好的选择。
# 但如果业务解释性和模型的简洁性对你更为重要，那么聚类数为3也可能是个合适的选择。


# In[ ]:




