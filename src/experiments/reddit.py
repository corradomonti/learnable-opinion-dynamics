import bz2

DATASET_PATH = "../../data/reddit/"

class AutoId:
    def __init__(self):
        self.next_id = 0
        self.map = dict()
    
    def __getitem__(self, key):
        i = self.map.get(key)
        if i is None:
            i = self.next_id
            self.map[key] = i
            self.next_id += 1
        return i

def timestamp2t(time_str):
    assert len(time_str) == 6
    year = int(time_str[:4])
    month = int(time_str[4:])
    t = (year - 2008) * 12 + (month - 1)
    return t

def _read_file(path):
    with bz2.open(path) as f:
        for line in f:
            time_str, i_name, j_name, w_str = line.split(b"\t")
            t = timestamp2t(time_str)
            yield t, i_name, j_name, int(w_str)

def read_reddit(T=120, T_test_min=None, T_test_max=None):
    print(f"Reading dataset from {DATASET_PATH}...")
    user2id = AutoId()
    u_v_t_weights = []
    u_v_test = []

    for t, u_name, v_name, w in _read_file(DATASET_PATH + "edges_user.tsv.bz2"):
        if t < T:
            u_v_t_weights.append((user2id[u_name], user2id[v_name], t, w))
    user2id = user2id.map
    
    if T_test_min is not None:
        for t, u_name, v_name, w in _read_file(DATASET_PATH + "edges_user.tsv.bz2"):
            if T_test_min <= t < T_test_max and u_name in user2id and v_name in user2id:
                u_v_test.append((user2id[u_name], user2id[v_name]))

    subreddit2id = AutoId()
    v_a_t_weights = []
    v_a_test = []
    
    for t, user_name, subreddit_name, w in _read_file(DATASET_PATH + "edges_feature.tsv.bz2"):
        if user_name in user2id and t < T:
            v_a_t_weights.append((user2id[user_name], subreddit2id[subreddit_name], t, w))
    subreddit2id = subreddit2id.map
    
    if T_test_min is not None:
        for t, user_name, subreddit_name, w in _read_file(DATASET_PATH + "edges_feature.tsv.bz2"):
            if T_test_min <= t < T_test_max and user_name in user2id and subreddit_name in subreddit2id:
                v_a_test.append((user2id[user_name], subreddit2id[subreddit_name]))
    
    print(len(user2id), "users,", len(subreddit2id), "features")
    print(len(u_v_t_weights), "user-user edges")
    print(len(v_a_t_weights), "user-feature edges")
    print(len(u_v_test), "test interactions,", len(v_a_test), "test node-features")
    
    return user2id, subreddit2id, u_v_t_weights, v_a_t_weights, u_v_test, v_a_test
