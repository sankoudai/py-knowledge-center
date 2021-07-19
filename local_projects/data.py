import pandas as pd
import pandasql as ps

# 训练数据
orders_train = pd.read_csv('dataset/orders_train.txt', sep='\t')
order_test = pd.read_csv('dataset/orders_test_poi.txt', sep='\t')

aor_recall = pd.read_csv('data/recs_aor.txt', sep='\t')
aor_recall['wm_poi_id'] = aor_recall['poi_ids'].str.split(',')
aor_recall = aor_recall.explode('wm_poi_id')
aor_recall['wm_poi_id'] = aor_recall['wm_poi_id'].astype('int32')
del aor_recall['poi_ids']

#
def get_recall(orders_tbl, recall_tbl):
    recall_sql = '''
        select
            orders_tbl.user_id,
            dim_tbl.aor_id,
            recall_tbl.wm_poi_id
        from orders_tbl
        inner join recall_tbl
        on orders_tbl.aor_id = recall_tbl.aor_id
    '''
