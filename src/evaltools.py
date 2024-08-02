import json
import matplotlib.pyplot as plt
from collections import Counter
import os
plt.rc("font", family="DejaVu Sans")

def compute_novelty(param) -> None:
    all_count = Counter()
    user_counts = []
    user_recs = []
    user_novelty = []
    
    with open(os.path.join(param.result_path, "predicted_titles.jsonl"), "r") as f:
        for line in f:
            uid, recs = zip(*json.loads(line.strip("\n")).items())
            recs = recs[0]
            user_count = {k: 1 for k in recs}
            user_counts.append((uid, user_count))
            user_recs.append((uid, len(recs)))
            all_count.update(user_count)
            
        user_recs = dict(user_recs)
        user_size = len(user_counts)
        for uid, count in user_counts:
            novelty_sum = 0
            for title in count:
                novelty = 1 - (all_count[title] / user_size)
                novelty_sum += novelty
            avg_novelty = novelty_sum / user_recs[uid]
            user_novelty.append((uid, avg_novelty))
            
    with open(os.path.join(param.result_path, "novelty.csv"), "w") as f:
        f.write("uid, avg_novelty\n")
        for uid, n in user_novelty:
            f.write(f"{uid[0]}, {n}\n")