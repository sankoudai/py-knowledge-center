# 给定aor_id， 召回如下poi:
# 1) 存在订单是poi送给到aor_id的
# 2）poi是aor_id的

import pandas as pd
import numpy as np

recall_dict = {}

# recall from orders
train_orders = pd.read_csv('dataset/orders_train.txt', sep='\t')
for i, row in train_orders.iterrows():
    poi_id = int(row['wm_poi_id'])
    aor_id = int(row['aor_id'])
    if aor_id not in recall_dict:
        recall_dict[aor_id] = set()
    recall_dict[aor_id].add(poi_id)

# recall from pois
pois = pd.read_csv('dataset/pois.txt', sep='\t')
pois = pois[np.logical_not(pois['aor_id'].isna())]
pois = pois[np.logical_not(pois['wm_poi_id'].isna())]

for i, row in pois.iterrows():
    poi_id = int(row['wm_poi_id'])
    aor_id = int(row['aor_id'])
    if aor_id not in recall_dict:
        recall_dict[aor_id] = set()
    recall_dict[aor_id].add(poi_id)

# write to files
with open('data/recs_aor.txt', 'w') as g:
    g.write('{}\t{}\n'.format('aor_id', 'poi_ids'))
    for aor_id, poi_ids in recall_dict.items():
        poi_str = ','.join([str(poi_id) for poi_id in poi_ids])
        g.write('{}\t{}\n'.format(aor_id, poi_str))

