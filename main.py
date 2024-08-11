# %%
import asyncio
import pandas as pd
import time
import sys
import os

sys.path.append("./")
from src.selection import select_users, select_item_splits
from src.prompt import Prompter
from src.recommender import RecParam
from src.dictionary import ItemDictionary
from src.pipeline import recommend_per_user
from src.evaltools import compute_novelty

#--------------------------------------
# Initial Setup
#--------------------------------------
with open("openai.key", "r") as f:
    os.environ["OPENAI_API_KEY"] = f.readlines()[0]

# For use with Replicate API
# with open("replicate.key", "r") as f:
#    os.environ["REPLICATE_API_TOKEN"] = f.readlines()[0]

hetrec = pd.read_csv("data/hetrec/user_ratedmovies.dat", sep="\t")
 # Important to adjust iid because this is how they are stored in the item dictionary
hetrec["movieID"] = hetrec["movieID"].apply(lambda x: int(x) - 1)

# Load parameters and item embeddings
param = RecParam.from_json(path="parameters.json")

item_dict = ItemDictionary.from_h5(
    path=f"./data/embeddings/hetrec_gpt3_embed_level4.h5"
)

if param.use_aux:
    item_dict.load_auxiliary("data/embeddings/aux_gpt3_embed_level3.h5")

item_dict.construct_quantile_map(param.min_sim_quantile)

#--------------------------------------
# Selection
#--------------------------------------
selected_users = select_users(
    hetrec, quantile_range=(0.50, 0.75), num_users=50, num_disliked_threshold=30
)

user_splits = select_item_splits(
    selected_users,
    example_size=param.num_examples,
    eval_size=param.eval_size,
    random_state=param.random_state,
)

prompter = Prompter(
    knowledge_cutoff=item_dict.knowledge_cutoff,
    k=param.k,
    final_k=param.final_k,
    allow_explanations=param.allow_explanations,
    prompt_style=param.prompt_style,
    allow_popular_movies=param.allow_popular_movies,
)

#--------------------------------------
# Start Experiment
#--------------------------------------
print("Starting...")
start = time.perf_counter()
asyncio.run(recommend_per_user(user_splits, item_dict, prompter, param))
print(f"Finished: {time.perf_counter() - start}")
compute_novelty(param)
