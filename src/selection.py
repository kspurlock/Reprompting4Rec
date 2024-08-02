from typing import Iterable, Union
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
import kmedoids
import copy


def select_users(
    data: pd.DataFrame,
    quantile_range: float,
    num_users: int,
    rating_threshold: float = 3.0,
    num_disliked_threshold: float = 10,
) -> dict[dict[Iterable]]:
    rating_table = data.pivot(
        index="userID", columns="movieID", values="rating"
    ).fillna(0)

    num_liked_order_index = (
        (rating_table >= rating_threshold).sum(axis=1).sort_values().index
    )

    rating_table = rating_table.loc[
        num_liked_order_index, :
    ]  # Reorder table descending based on number of liked items

    liked_table = rating_table >= rating_threshold
    num_liked = liked_table.sum(axis=1)  # Recount how many items each user has liked

    disliked_table = ~liked_table & (rating_table > 0)
    num_disliked = disliked_table.sum(axis=1)

    num_liked_lower_threshold = np.quantile(num_liked, quantile_range[0])
    num_liked_upper_threshold = np.quantile(num_liked, quantile_range[1])

    user_mask = (
        (num_liked_upper_threshold >= num_liked)
        * (num_liked >= num_liked_lower_threshold)
        * (num_disliked >= num_disliked_threshold)
    )

    selected_users = rating_table.loc[user_mask]

    dataset_dict = defaultdict(dict)
    for i, (uid, row) in enumerate(selected_users.iterrows()):
        liked_mask = row >= rating_threshold
        disliked_mask = ~liked_mask * (row > 0)

        dataset_dict[uid]["like"] = [
            *zip(row.index[liked_mask], row.values[liked_mask])
        ]
        dataset_dict[uid]["dislike"] = [
            *zip(row.index[disliked_mask], row.values[disliked_mask])
        ]

        if num_users <= i + 1:
            break

    return dataset_dict


def split_try_stratify(
    X: Iterable[tuple], train_size: float, random_state: int
) -> tuple[Iterable, Iterable]:
    _, rating_scores = zip(*X)
    score_class, score_counts = np.unique(rating_scores, return_counts=True)

    # Cases where stratify can't be used:
    if (score_counts == 1).any():
        # Only one sample from class
        stratify = None
    elif train_size < len(score_class):
        # Train size < number of classes
        stratify = None
    else:
        # OK to use stratify
        stratify = rating_scores

    train_split, test_split = train_test_split(
        X,
        train_size=train_size,
        stratify=stratify,
        random_state=random_state,
    )
    return train_split, test_split


def example_feedback_eval_split(
    user_rated_items: Iterable,
    example_size: Union[int, float],
    eval_size: float,
    random_state: int,
) -> dict[Iterable]:
    size = len(user_rated_items)

    if size == 0:
        return {"example": (), "feedback": (), "eval": ()}

    elif 0 < size < 3:
        iids, _ = zip(*user_rated_items)
        return {"example": iids, "feedback": iids, "eval": iids}

    ####################################
    # ADJUSTING SPLIT SIZES IF NECESSARY
    ####################################

    if isinstance(
        example_size, float
    ):  # If ratio, convert to a fixed number of samples
        example_size_ = int((size * example_size))
    else:
        example_size_ = example_size

    if isinstance(eval_size, float):
        eval_size_ = int((size * eval_size))
    else:
        eval_size_ = eval_size

    feedback_size_ = size - example_size_ - eval_size_

    if min([example_size_, feedback_size_, eval_size_]) <= 0:
        # If current split sizes would reduce any of the sets to zero
        example_size_ = int(size / 3)
        feedback_size_ = int((size - example_size_) / 2)

    ###################################
    # BEGINNING SPLITTING
    ###################################

    example_set, feedback_and_eval_set = split_try_stratify(
        user_rated_items, train_size=example_size_, random_state=random_state
    )

    feedback_set, eval_set = split_try_stratify(
        feedback_and_eval_set, train_size=feedback_size_, random_state=random_state
    )

    ###################################
    # PACKAGING AND RETURN
    ###################################
#    example_iids, example_ratings = zip(*example_set)
#    feedback_iids, feedback_ratings = zip(*feedback_set)
#    eval_iids, eval_ratings = zip(*eval_set)

    packaged = {"example": example_set, "feedback": feedback_set, "eval": eval_set}
    return packaged


def select_item_splits(
    user_selections: dict[dict[Iterable]],
    example_size: Union[int, float],
    eval_size: float,
    random_state: int,
) -> dict[dict[dict[Iterable]]]:
    user_splits = {uid: {"like": None, "dislike": None} for uid in user_selections}

    for uid, rated_items in user_selections.items():
        liked = rated_items["like"]
        user_splits[uid]["like"] = example_feedback_eval_split(
            liked, example_size, eval_size, random_state=random_state
        )

        disliked = rated_items["dislike"]
        user_splits[uid]["dislike"] = example_feedback_eval_split(
            disliked,
            example_size,
            eval_size,
            random_state=random_state,
        )

    return user_splits


#def select_kmedoids_splits(
#    user_splits: dict, item_dict, k: int, random_state: int = None
#):
#    medoid_user_splits = copy.deepcopy(user_splits)

#    for uid, splits in medoid_user_splits.items():
#        for sentiment in splits:  # like/dislike
#            for set_type in splits[sentiment]:  # example/feedback/eval
#                item_set = splits[sentiment][set_type]
#                if len(item_set) <= k:
#                    pass
#                else:
#                    embeddings = item_dict.get_embedding_from_iid(item_set)
#                    diss_matrix = 1 - (
#                        embeddings @ embeddings.T
#                    )  # Maybe need to scale this?
#                    medoid_indices = kmedoids.fasterpam(
#                        diss_matrix, medoids=k, random_state=random_state
#                    ).medoids
#                    medoid_iids = np.array(item_set)[medoid_indices]
#                    medoid_user_splits[uid][sentiment][set_type] = tuple(medoid_iids)

#    return medoid_user_splits

def select_kmedoid_splits(
    user_splits: dict, item_dict, random_state: int = None
    ):
    medoid_user_splits = copy.deepcopy(user_splits)
    for uid, splits in medoid_user_splits.items():
        for sentiment in splits:
            for set_type in splits[sentiment]:
                iids, ratings = zip(*splits[sentiment][set_type])
                rating_values, counts = np.unique(ratings, return_counts=True)
                
                embeddings = item_dict.get_embedding_from_iid(iids)
                iids_ = np.array(iids)
                ratings_ = np.array(ratings)
                
                new_set = []
                for i, r in enumerate(rating_values):
                    ratio = 1/len(counts)
                    if (counts[i] > (int(counts.sum() * ratio))) and (counts.sum() > 10):
                        diss_matrix = 1 - (
                            embeddings[ratings == r] @ embeddings[ratings == r].T
                        )
                        medoid_indices = kmedoids.fasterpam(
                            diss_matrix, medoids=int(counts.sum() * ratio), random_state=random_state
                        ).medoids
                        
                        medoid_iids = iids_[ratings == r][medoid_indices]
                        new_set += list(zip(medoid_iids, ratings_[ratings == r]))
                    else:
                        new_set += list(zip(iids_[ratings == r], ratings_[ratings == r]))
                        
                medoid_user_splits[uid][sentiment][set_type] = new_set
                
    return medoid_user_splits


if __name__ == "__main__":
    from dictionary import dict_from_h5
    import sys

    sys.path.append("../embeddings")
    sys.path.append("../data")

    hetrec = pd.read_csv("data/hetrec/user_ratedmovies.dat", sep="\t")
    hetrec["movieID"] = hetrec["movieID"].apply(lambda x: int(x) - 1)

    selected_users = select_users(
        hetrec, quantile_range=(0.25, 0.50), num_users=50, num_disliked_threshold=25
    )

    user_splits = select_item_splits(
        selected_users, 10, eval_size=1 / 3, random_state=192
    )

    item_dict = dict_from_h5(path="data/embeddings/hetrec_gpt3_embed.h5", quantile=0.85)

    #select_kmedoids_splits(user_splits, item_dict, k=10, random_state=192)
