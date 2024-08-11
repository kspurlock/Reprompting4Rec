import copy
from typing import Iterable, Union, Optional, Dict
import numpy as np


class User:
    def __init__(
        self,
        uid: dict,
        iid_dict: dict,
        title_dict: dict,
        rating_dict: dict,
        embedding_dict: Dict[str, np.ndarray],
    ):
        self.uid = uid
        self.iids = iid_dict
        self.titles = title_dict
        self.ratings = rating_dict
        self.embeddings = embedding_dict

    def get_sizes(self) -> Iterable[int]:
        lengths = [
            len(self.iids["like"]["example"]),
            len(self.iids["like"]["feedback"]),
            len(self.iids["like"]["eval"]),
            len(self.iids["dislike"]["example"]),
            len(self.iids["dislike"]["feedback"]),
            len(self.iids["dislike"]["eval"]),
        ]

        return lengths


def construct_user(uid: int, user_split: dict, item_dict):
    iid_dict = copy.deepcopy(user_split)
    title_dict = copy.deepcopy(
        user_split
    )  # Only copying these to keep the same nested structure, will replace
    rating_dict = copy.deepcopy(user_split)
    embedding_dict = copy.deepcopy(user_split)

    for sentiment, sets in user_split.items():
        for set_type, item_ratings in sets.items():
            iids, ratings = zip(*item_ratings)
            iid_dict[sentiment][set_type] = iids
            title_dict[sentiment][set_type] = item_dict.get_title_from_iid(iids)
            rating_dict[sentiment][set_type] = ratings
            embedding_dict[sentiment][set_type] = np.array(
                item_dict.get_embedding_from_iid(iids)
            )

    user = User(uid, iid_dict, title_dict, rating_dict, embedding_dict)
    return user


class SetContainer:
    """Primarily defined to track coverage/recall on the liked set over prompts"""

    def __init__(
        self,
        titles: Iterable[str],
        ratings: Iterable[float],
        embeddings: np.ndarray,
        quantiles: Iterable[float],
    ):
        self.titles = titles
        self.ratings = ratings
        self.embeddings = embeddings
        self.quantiles = quantiles
        self.map = {k: [] for k in titles}
        self.covered = 0

    def is_empty(self, title_index: int):
        return len(self.map[self.titles[title_index]]) == 0

    def add_item(self, title_index: int, add_item: tuple[str, float]) -> None:
        if self.is_empty(title_index):
            self.covered += 1
        self.map[self.titles[title_index]].append(add_item)

    def get_recall(self):
        return self.covered / len(self.titles)


def construct_set(user: User, sentiment: str, set_type: str, item_dict) -> SetContainer:
    set_container = SetContainer(
        user.titles[sentiment][set_type],
        user.ratings[sentiment][set_type],
        user.embeddings[sentiment][set_type],
        item_dict.get_quantile_from_iid(user.iids[sentiment][set_type]),
    )

    return set_container


class Evaluator:
    def __init__(self):
        self.all_predicted_titles = []

        # Statistics
        self.recall_at_p = []
        self.precision_at_p = []
        self.ndcg_at_p = []
        self.ils_at_p = []
        self.ap_at_p = []
        self.covered = 0
        self.unmatched_count = 0
        self.unmatched_titles = []

    def _compute_ndcg(
        self,
        relevant: np.ndarray,
    ) -> float:
        discounts = np.log2(np.arange(2, len(relevant) + 2))
        idcg = np.sum((np.power(2, (relevant >= 0)) - 1) / discounts)
        dcg = np.sum((np.power(2, (relevant == 1)) - 1) / discounts)

        if (idcg == 0).any():
            ndcg = 0
        else:
            ndcg = dcg / idcg
        return ndcg

    def _compute_ILS(self, pred_embeddings: np.ndarray) -> float:
        sim = pred_embeddings @ pred_embeddings.T
        tri = np.tril(sim)
        np.fill_diagonal(tri, 0)
        sum_of_sim = np.sum(tri)
        denom = (len(tri) * (len(tri) - 1)) / 2
        return (sum_of_sim / denom) if denom != 0 else 0

    def _compute_AP(self, relevant: np.ndarray) -> float:
        valid = relevant[np.nonzero(relevant >= 0)]
        p = []
        for i in range(0, len(valid)):
            p.append(np.mean(valid[: i + 1]))
        ap = np.mean(p)
        return ap

    def evaluate_matching(
        self,
        liked_set: SetContainer,  # One of feedback/eval
        disliked_set: SetContainer,
        pred_embeddings: np.ndarray,
        pred_titles: Iterable[str],
    ) -> Iterable[str]:
        like_sim_mat = (
            pred_embeddings @ liked_set.embeddings.T
        )  # K x 1536 * 1536 x N = K x N
        dislike_most_similar = np.max(
            pred_embeddings @ disliked_set.embeddings.T, axis=1
        )  # K x 1

        relevant = np.zeros(
            len(pred_titles)
        )  # Vector to mark whether a prediction was relevant/matched

        for i, (like_sim_vec, dislike_item_sim) in enumerate(
            zip(like_sim_mat, dislike_most_similar)
        ):
            for j, sim in enumerate(like_sim_vec):
                if (
                    sim <= 0
                ):  # Case where item can't be matched, don't penalize but mark with -1
                    relevant[i] = -1
                    self.unmatched_count += 1
                    self.unmatched_titles.append(pred_titles[i])
                    break
                elif sim > dislike_item_sim and sim >= liked_set.quantiles[j]:
                    relevant[i] = 1
                    liked_set.add_item(title_index=j, add_item=(pred_titles[i], sim))

        recall = liked_set.get_recall()
        precision = (
            np.sum(relevant == 1) / (np.sum(relevant >= 0))
            if np.sum(relevant >= 0) > 0
            else 0
        )
        ndcg = self._compute_ndcg(relevant)
        ils = self._compute_ILS(pred_embeddings[relevant >= 0])
        ap = self._compute_AP(relevant)

        self.recall_at_p.append(recall)
        self.precision_at_p.append(precision)
        self.ndcg_at_p.append(ndcg)
        self.ils_at_p.append(ils)
        self.ap_at_p.append(ap)

        self.all_predicted_titles += pred_titles

        irrelevant_titles = np.array(pred_titles)[(relevant == 0)]
        relevant_titles = np.array(pred_titles)[(relevant == 1)]

        return irrelevant_titles, relevant_titles

    def evaluate_weighted(
        self,
        liked_set: SetContainer,  # One of feedback/eval
        disliked_set: SetContainer,
        pred_embeddings: np.ndarray,
        pred_titles: Iterable[str],
    ) -> Iterable[str]:
        l_embeddings = liked_set.embeddings
        l_ratings = liked_set.ratings
        l_quantiles = liked_set.quantiles
        d_embeddings = disliked_set.embeddings
        d_ratings = disliked_set.ratings
        d_quantiles = disliked_set.quantiles

        combined_embeddings = np.vstack((l_embeddings, d_embeddings))
        combined_ratings = np.array(l_ratings + d_ratings)
        combined_quantiles = np.array(l_quantiles + d_quantiles)

        try:
            like_sim_mat = (
                pred_embeddings @ liked_set.embeddings.T
            )  # K x 1536 * 1536 x N = K x N
            dislike_most_similar = np.max(
                pred_embeddings @ disliked_set.embeddings.T, axis=1
            )  # K x 1

        except ValueError:
            return [""], [""]

        base_relevant = np.zeros(
            len(pred_titles)
        )  # Vector to mark whether a prediction was relevant/matched

        for i, (like_sim_vec, dislike_item_sim) in enumerate(
            zip(like_sim_mat, dislike_most_similar)
        ):
            for j, sim in enumerate(like_sim_vec):
                if (
                    sim <= 0
                ):  # Case where item can't be matched, don't penalize but mark with -1
                    base_relevant[i] = -1
                    self.unmatched_count += 1
                    self.unmatched_titles.append(pred_titles[i])
                    break
                elif sim > dislike_item_sim and sim >= liked_set.quantiles[j]:
                    base_relevant[i] = 1
                    liked_set.add_item(title_index=j, add_item=(pred_titles[i], sim))

        ws = compute_weighted_sim(
            pred_embeddings, combined_embeddings, combined_ratings, combined_quantiles
        )
        ws_relevant = ws >= 3

        full_relevant = ws_relevant * base_relevant

        recall = liked_set.get_recall()
        precision = (
            np.sum(full_relevant == 1) / (np.sum(full_relevant >= 0))
            if np.sum(full_relevant >= 0) > 0
            else 0
        )
        ndcg = self._compute_ndcg(full_relevant)
        ils = self._compute_ILS(pred_embeddings[full_relevant >= 0])
        ap = self._compute_AP(full_relevant)

        self.recall_at_p.append(recall)
        self.precision_at_p.append(precision)
        self.ndcg_at_p.append(ndcg)
        self.ils_at_p.append(ils)
        self.ap_at_p.append(ap)

        self.all_predicted_titles += pred_titles

        irrelevant_titles = np.array(pred_titles)[(full_relevant == 0)]
        relevant_titles = np.array(pred_titles)[(full_relevant == 1)]

        return irrelevant_titles, relevant_titles

    def evaluate(
        self,
        liked_set: SetContainer,  # One of feedback/eval
        disliked_set: SetContainer,
        pred_embeddings: np.ndarray,
        pred_titles: Iterable[str],
        eval_style: str,
    ) -> Iterable[str]:
        if eval_style == "matching":
            irrelevant_titles, relevant_titles = self.evaluate_matching(
                liked_set, disliked_set, pred_embeddings, pred_titles
            )

        elif eval_style == "weighted":
            irrelevant_titles, relevant_titles = self.evaluate_weighted(
                liked_set, disliked_set, pred_embeddings, pred_titles
            )

        else:
            raise ValueError(
                "Invalid argument for 'evaluation_style'. Did you mean matching or weighted?"
            )

        return irrelevant_titles, relevant_titles


def compute_weighted_sim(
    pred_embeddings, true_embeddings, true_ratings, combined_quantiles
):
    pred_sim = pred_embeddings @ true_embeddings.T
    top_n_indices = np.argsort(pred_sim, axis=1)[:, ::-1]
    nn_sim = np.take_along_axis(pred_sim, top_n_indices, axis=1)

    valid = nn_sim > combined_quantiles

    ratings_mat = np.tile(true_ratings[:, np.newaxis], len(pred_embeddings)).T
    nn_ratings = np.take_along_axis(ratings_mat, top_n_indices, axis=1)

    sum_weighted_sim = np.sum(nn_sim * valid * nn_ratings, axis=1)
    sum_sim = np.sum(nn_sim * valid, axis=1)

    return np.divide(sum_weighted_sim, sum_sim, where=sum_sim != 0)
