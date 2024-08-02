from rapidfuzz.distance import Levenshtein
from typing import Iterable, Union
import re
import copy
import pandas as pd
import numpy as np
import h5py


def parenth_processor(string: str, remove_year: bool = False, remove_alt: bool = True):
    new_str = string
    matches = re.findall(r"( \([^\)]+\))", new_str)
    if matches:
        if len(matches) == 1 and remove_year:
            return new_str.replace(matches[0], "")

        elif len(matches) == 2:
            if remove_alt:
                new_str = new_str.replace(matches[0], "")

            if remove_year:
                new_str = new_str.replace(matches[1], "")

    return new_str


def article_swapper(title: str):
    """Moves parts of the titles like 'The' and 'A' back to the beginning of the title."""
    new_title = title
    if re.search(r", [\w']* \(", title):
        new_title = (
            re.findall(r", [\w']*", title)[0][2:] + " " + re.sub(r", [\w']*", "", title)
        )

    return new_title


def title_processor(
    string: str, remove_year: bool = False, remove_alt: bool = True, lower: bool = True
):
    new_title = article_swapper(string)
    new_title = parenth_processor(new_title, remove_year, remove_alt)

    return new_title.lower() if lower else new_title


def norm_lev_sim(s1: str, s2: str, processor: callable, weights=None):
    s1_ = s1.lower()
    s2_ = s2.lower()

    weights_ = weights or (1, 1, 1)

    dist = Levenshtein.distance(s1_, s2_, processor=processor, weights=weights_)
    alpha = max(weights_[:2])

    return 1 - (2 * dist) / ((alpha * (len(s1) + len(s2))) + dist)


class ItemDictionary:
    def __init__(
        self,
        item_ids: Iterable[int],
        years: Iterable[int],
        title_years: Iterable[str],
        item_embeddings: np.ndarray,
    ):
        self._iid_to_title = dict(
            zip(item_ids, title_years)
        )  # Titles are in "Name (Year)" format
        self._iid_to_year = dict(zip(item_ids, years))
        self._title_to_iid = dict(zip(title_years, item_ids))
        self._iid_to_embedding = dict(zip(item_ids, item_embeddings))
        self._iid_to_quantile = None  # Has to be manually constructed
        self._internal_embed_dim = item_embeddings.shape[1]

        self.knowledge_cutoff = np.max(years)
        # self.similarity_scale = compute_similarity_scale(item_embeddings)

    @property
    def embeddings(self) -> np.ndarray:
        embed = np.array(list(self._iid_to_embedding.values()))
        return embed

    def get_title_from_iid(self, iids: Union[int, Iterable[int]]) -> Iterable[str]:
        if not isinstance(iids, Iterable):
            iids_ = [iids]
        else:
            iids_ = iids

        return list(map(self._iid_to_title.__getitem__, iids_))

    def get_iid_from_title(self, titles: Union[str, Iterable[str]]) -> Iterable[int]:
        if isinstance(titles, str):
            titles_ = [titles]
        else:
            titles_ = titles

        return list(map(self._title_to_iid.__getitem__, titles_))

    def get_quantile_from_iid(self, iids: Union[int, Iterable[int]]) -> Iterable[int]:
        if isinstance(iids, str):
            iids_ = [iids]
        else:
            iids_ = iids

        return list(map(self._iid_to_quantile.__getitem__, iids_))

    def get_year_from_iid(self, iids: Union[int, Iterable[int]]) -> Iterable[str]:
        if not isinstance(iids, Iterable):
            iids_ = [iids]
        else:
            iids_ = iids

        return list(map(self._iid_to_year.__getitem__, iids_))

    def get_embedding_from_iid(self, iid: Union[int, Iterable[int]]) -> np.ndarray:
        if isinstance(iid, Iterable):
            embed = np.array([self._iid_to_embedding[i] for i in iid])
        else:
            embed = np.array([self._iid_to_embedding[iid]])
        return embed

    def get_all_titles(self) -> list[str]:
        return list(self._title_to_iid.keys())

    def get_all_iids(self) -> list[int]:
        return list(self._iid_to_title.keys())

    def get_embedding_from_title(
        self,
        titles: Union[str, Iterable[str]],
        title_threshold: float,
        verbose: bool = False,
    ) -> np.ndarray:
        if isinstance(titles, str):
            titles_ = [titles]
        else:
            titles_ = titles

        embeddings = []
        for title in titles_:
            # Exact Lookup
            try:
                embedding = self._iid_to_embedding[self._title_to_iid[title]]
            except KeyError:
                print(
                    f"Exact lookup for: '{title}' failed, trying fuzzy lookup..."
                ) if verbose else None
                embedding = self._fuzzy_lookup(title, title_threshold, verbose)

            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        return embeddings

    def _fuzzy_lookup(self, title: str, threshold: float, verbose: bool) -> np.ndarray:
        scores = [
            (
                key,
                norm_lev_sim(title, key, processor=title_processor, weights=(1, 1, 3)),
            )
            for key in self._title_to_iid
        ]

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        print(
            f"Closest title is: {scores[0][0]} with score: {scores[0][1]}"
        ) if verbose else None

        if scores[0][1] > threshold:
            return self._iid_to_embedding[self._title_to_iid[scores[0][0]]]

        else:
            print(
                f"Title similarity < {threshold}, returning zero vector"
            ) if verbose else None
            return np.zeros((self._internal_embed_dim,))

    def construct_quantile_map(self, quantile: float) -> None:
        iids = self.get_all_iids()
        item_embeddings = self.embeddings

        sim = item_embeddings @ item_embeddings.T
        quantile_values = np.quantile(sim, axis=1, q=quantile)
        self._iid_to_quantile = dict(zip(iids, quantile_values))
        return

    def load_auxiliary(self, path: str) -> None:
        """Loads auxiliary (commonly unmatched) dataset to dictionary"""

        with h5py.File(path, "r") as handle:
            max_id = max(self.get_all_iids())
            years = handle["ReleaseYear"][:]
            titles = [i.decode("utf-8") for i in handle["MovieTitle"][:]]
            title_years = [title + f" ({year})" for title, year in zip(titles, years)]
            item_embeddings = handle["Embedding"][:]
            item_ids = np.array([f"{max_id + i}" for i in range(len(item_embeddings))])

            new_iid_to_embedding = {i: e for i, e in zip(item_ids, item_embeddings)}
            new_iid_to_year = {i: y for i, y in zip(item_ids, years)}
            new_title_to_iid = {t: i for t, i in zip(title_years, item_ids)}

            self._iid_to_embedding.update(new_iid_to_embedding)
            self._iid_to_year.update(new_iid_to_year)
            self._title_to_iid.update(new_title_to_iid)
    


def dict_from_h5(
    path: str,
) -> ItemDictionary:
    with h5py.File(path, "r") as handle:
        item_ids = handle["MovieID"][:] - 1
        years = handle["ReleaseYear"][:]
        titles = [i.decode("utf-8") for i in handle["MovieTitle"][:]]
        title_years = [title + f" ({year})" for title, year in zip(titles, years)]
        item_embeddings = handle["Embedding"][:]

    item_dict = ItemDictionary(item_ids, years, title_years, item_embeddings)
    return item_dict
