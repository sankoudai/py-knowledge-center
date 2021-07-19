from recaller_base import Recaller
import pandas as pd

class AorRecaller(Recaller):
    def __init__(self, aor_rec_files):
        self.aor_rec_files = aor_rec_files
        self.recall_dict = {}
        self._setup()

    def recall(self, user_id=None, aor_id=None):
        if aor_id and aor_id in self.recall_dict:
            return self.recall_dict.get(aor_id)
        return set()

    def _setup(self):
        for aor_rec_file in self.aor_rec_files:
            df = pd.read_csv(aor_rec_file, sep='\t')
            for i, row in df.iterrows():
                aor_id = row['aor_id']
                poi_ids = [int(poi_id) for poi_id in row['poi_ids'].split(',')]
                if aor_id not in self.recall_dict:
                    self.recall_dict[aor_id] = set()
                self.recall_dict[aor_id].update(poi_ids)
