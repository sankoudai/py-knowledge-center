#特征列: dt,wm_poi_id,feature

import pandas as pd
from pandasql import sqldf

global_sqldf = lambda q: sqldf(q, globals())

# 加载数据
df = pd.read_csv('dataset/orders_train.txt', sep='\t')

feature_sql = '''
    select
        wm_poi_id,
        count(1) order_cnt
    from df
    group by wm_poi_id
'''
feature_df = global_sqldf(feature_sql)

feature_df.to_csv('data/feature_poi.txt', sep='\t', index=False)