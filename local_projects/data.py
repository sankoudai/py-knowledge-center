import pandas as pd
import pandasql as ps

# 原始数据
orders_train = pd.read_csv('dataset/orders_train.txt', sep='\t')
order_test = pd.read_csv('dataset/orders_test_poi.txt', sep='\t')
order_test['wm_poi_id'] = -1

aor_recall = pd.read_csv('data/recs_aor.txt', sep='\t')
aor_recall['wm_poi_id'] = aor_recall['poi_ids'].str.split(',')
aor_recall = aor_recall.explode('wm_poi_id')
aor_recall['wm_poi_id'] = aor_recall['wm_poi_id'].astype('int32')
del aor_recall['poi_ids']

pd.merge(orders_train, aor_recall, on='aor_id')
# 关联
def get_recall(orders_tbl, recall_tbl):
    recall_sql = '''
        select
            order_tbl.dt,
            order_tbl.user_id,
            order_tbl.wm_order_id,
            order_tbl.aor_id,
            order_tbl.order_timestamp,
            order_tbl.ord_period_name,
            order_tbl.aoi_id,
            order_tbl.takedlvr_aoi_type_name,
            recall_tbl.wm_poi_id,
            if(orders_tbl.wm_poi_id==recall_tbl.wm_poi_id, 1, 0) label
        from orders_tbl
        inner join recall_tbl
        on orders_tbl.aor_id = recall_tbl.aor_id
    '''
    return ps.sqldf(recall_sql, locals())

train_data = get_recall(orders_train, aor_recall)
train_data.head().to_csv('data/train_data.txt', sep='\t', index=False)
