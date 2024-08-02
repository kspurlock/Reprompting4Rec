from typing import Iterable
import numpy as np


class Prompter:
    def __init__(
        self,
        knowledge_cutoff: int,
        k: int,
        final_k: int,
        allow_explanations: bool,
        prompt_style: str,
        allow_popular_movies: bool
    ):
        self.knowledge_cutoff = knowledge_cutoff
        self.k = k
        self.final_k = final_k
        self.allow_explanations = allow_explanations
        self.prompt_style = prompt_style
        self.allow_popular_movies = allow_popular_movies

    def build_zeroshot_prompt(
        self,
        like_titles: Iterable[str],
        dislike_titles: Iterable[str],
    ) -> str:
        """Constructs an initial prompt to provide ChatGPT using training titles for a user. Only to be used at the initial consruction of a recommender object

        Args:
            train_titles (Iterable[str]): List of titles randomly selected from the overall set of user relevant ratings
            item_dict (ItemDictionary): ItemDictionary object (only used for the computed knowledge cutoff)
            k (int): Number of predictions to request from the system

        Returns:
            str: A string input prepared for the initial prompt
        """
        like_title_string = ":like, ".join(like_titles)
        dislike_title_string = ":dislike, ".join(dislike_titles)

        input_string = (
            "A user liked the following movies: "
            + like_title_string
            + ".\nThe user disliked the following movies: "
            + dislike_title_string
            + f".\nBased on the user's likes and dislikes, you must recommend {self.k} other movies"
            + f" released before {self.knowledge_cutoff} that they may also enjoy."
            + (f"\nTry to recommend movies that are less popular." if not self.allow_popular_movies else "")
            + "\nSort your recommendations by your confidence that the user will like the movie and "
            + (
                "provide a short explanation for each recommendation."
                if self.allow_explanations
                else "do not provide explanations for each recommendation."
            )
        )

        return input_string

#     def _get_fewshot_examples(self, provide_reasoning, random_state, item_dict):
#         rand_gen = np.random.default_rng(random_state)

#         all_iids = item_dict.get_all_iids()
#         fake_like_iids = rand_gen.choice(all_iids, self.k)
#         remaining_iids = np.setdiff1d(all_iids, fake_like_iids)

#         fake_dislike_iids = rand_gen.choice(remaining_iids, self.k)
#         remaining_iids = np.setdiff1d(remaining_iids, fake_dislike_iids)

#         # Use these to set up possible recommendations
#         remaining_embeddings = item_dict.get_embedding_from_iid(remaining_iids)
#         remaining_titles = item_dict.get_title_from_iid(remaining_iids)

#         # Like information
#         fake_like_titles = item_dict.get_title_from_iid(fake_like_iids)
#         fake_like_embeddings = item_dict.get_embedding_from_iid(fake_like_iids)
#         fake_like_quantiles = item_dict.get_quantile_from_iid(fake_like_iids)

#         # Dislike information
#         fake_dislike_titles = item_dict.get_title_from_iid(fake_dislike_iids)
#         fake_dislike_embeddings = item_dict.get_embedding_from_iid(fake_dislike_iids)

#         like_sim_mat = fake_like_embeddings @ remaining_embeddings.T
#         ordered_like_sim_mat = np.argsort(like_sim_mat, axis=1)
#         dislike_most_similar = np.max(
#             fake_like_embeddings @ fake_dislike_embeddings.T, axis=1
#         )
#         dislike_least_similar = np.argmin(
#             fake_like_embeddings @ fake_dislike_embeddings.T, axis=1
#         )

#         fake_rec = []
#         reasons = []
#         for i, (like_sim_idx_vec, dislike_item_sim) in enumerate(
#             zip(ordered_like_sim_mat, dislike_most_similar)
#         ):
#             for col_idx in like_sim_idx_vec[::-1]:  # Iterate descending
#                 if (
#                     like_sim_mat[i][col_idx] > dislike_item_sim
#                     and like_sim_mat[i][col_idx] >= fake_like_quantiles[i]
#                 ):
#                     valid_title = remaining_titles[col_idx]
#                     if provide_reasoning:
#                         reason = f"Because the user likes {fake_like_titles[i]} and dislikes {fake_dislike_titles[dislike_least_similar[i]]}, I would recommend:\n"
#                         reasons.append(reason)
#                     fake_rec.append(valid_title)
#                     break

#         formatted_fake_rec = [f"{i+1}. {title}" for i, title in enumerate(fake_rec)]

#         if provide_reasoning:
#             fake_rec_string = "".join(
#                 [reasons[i] + rec + "\n" for i, rec in enumerate(formatted_fake_rec)]
#             )
#         else:
#             fake_rec_string = "\n".join(formatted_fake_rec)

#         return fake_like_titles, fake_dislike_titles, fake_rec_string
    
    def _get_fewshot_examples(self, provide_reasoning, random_state, item_dict):
        rand_gen = np.random.default_rng(random_state)

        all_iids = item_dict.get_all_iids()
        fake_like_iids = rand_gen.choice(all_iids, self.k)
        remaining_iids = np.setdiff1d(all_iids, fake_like_iids)

        fake_dislike_iids = rand_gen.choice(remaining_iids, self.k)
        remaining_iids = np.setdiff1d(remaining_iids, fake_dislike_iids)

        # Use these to set up possible recommendations
        remaining_embeddings = item_dict.get_embedding_from_iid(remaining_iids)
        remaining_titles = item_dict.get_title_from_iid(remaining_iids)

        # Like information
        fake_like_titles = item_dict.get_title_from_iid(fake_like_iids)
        fake_like_embeddings = item_dict.get_embedding_from_iid(fake_like_iids)
        fake_like_quantiles = item_dict.get_quantile_from_iid(fake_like_iids)

        # Dislike information
        fake_dislike_titles = item_dict.get_title_from_iid(fake_dislike_iids)
        fake_dislike_embeddings = item_dict.get_embedding_from_iid(fake_dislike_iids)

        like_sim_mat = fake_like_embeddings @ remaining_embeddings.T
        ordered_like_sim_mat = np.argsort(like_sim_mat, axis=1)
        dislike_most_similar = np.max(
            fake_like_embeddings @ fake_dislike_embeddings.T, axis=1
        )
        dislike_least_similar = np.argmin(
            fake_like_embeddings @ fake_dislike_embeddings.T, axis=1
        )

        selected_indices = []
        selected_titles = []
        selected_like_titles = []
        selected_dislike_titles = []
        
        # Find titles for recommended, like, dislike
        for i, (like_sim_idx_vec, dislike_item_sim) in enumerate(
            zip(ordered_like_sim_mat, dislike_most_similar)
        ):
            for col_idx in like_sim_idx_vec[::-1][2:]:  # Iterate descending
                if (
                    like_sim_mat[i][col_idx] > dislike_item_sim
                    and like_sim_mat[i][col_idx] >= fake_like_quantiles[i]
                ):
                    selected_indices.append(col_idx)
                    selected_titles.append(remaining_titles[col_idx])
                    selected_like_titles.append(fake_like_titles[i])
                    selected_dislike_titles.append(fake_dislike_titles[dislike_least_similar[i]])
                    break

        # Sorting based on relevancy to initial list
        selected_embeddings = remaining_embeddings[selected_indices, :]
        sum_of_sim = (selected_embeddings @ fake_like_embeddings.T).sum(axis=1)
        sorted_sum = np.argsort(sum_of_sim)
                    
        sorted_selected_titles = np.array(selected_titles)[sorted_sum]
        sorted_like_titles = np.array(selected_like_titles)[sorted_sum]
        sorted_dislike_titles = np.array(selected_dislike_titles)[sorted_sum]
        
        # Construct parts of the prompt
        fake_rec = []
        reasons = []
        for s, l, d in zip(sorted_selected_titles, sorted_like_titles, sorted_dislike_titles):
            valid_title = s
            if provide_reasoning:
                reason = f"Because the user likes {l} and dislikes {d}, I would recommend:\n"
                reasons.append(reason)
            fake_rec.append(valid_title)
            
        # Format as a numbered list
        formatted_fake_rec = [f"{i+1}. {title}" for i, title in enumerate(fake_rec)]
        
        if provide_reasoning:
            fake_rec_string = "".join([reasons[i] + rec + "\n" for i, rec in enumerate(formatted_fake_rec)])
        else:
            fake_rec_string = "\n".join(formatted_fake_rec)
        
        fake_rec_string += "These recommendations are all sorted in descending order based on similarity to the user's preferences."

        return fake_like_titles, fake_dislike_titles, fake_rec_string

    def build_fewshot_prompt(
        self,
        like_titles: Iterable[str],
        dislike_titles: Iterable[str],
        provide_reasoning: bool,
        random_state: int,
        item_dict,
    ) -> str:
        base_prompt = "Your task is to recommend movies that a user might enjoy based on their likes and dislikes. Here is an example:"

        (
            fake_like_titles,
            fake_dislike_titles,
            fake_rec_string,
        ) = self._get_fewshot_examples(
            provide_reasoning, random_state, item_dict
        )

        ex_task = "\nTask: " + self.build_zeroshot_prompt(
            like_titles=fake_like_titles, dislike_titles=fake_dislike_titles
        )
        ex_response = "\nResponse:\n" + fake_rec_string

        actual_task_notify = "Given this example, complete the following task:\n"

        actual_task = self.build_zeroshot_prompt(
            like_titles=like_titles, dislike_titles=dislike_titles
        )

        actual_response = "\nResponse:\n"

        full = (
            base_prompt
            + ex_task
            + ex_response
            + actual_task_notify
            + actual_task
            + actual_response
        )

        return full

    def build_next_prompt(self, invalid_recs: Iterable[str], valid_recs: Iterable[str], retry: bool) -> str:
        if len(invalid_recs) > 0:
            # Handles the case where there is at least 1 non-similar recommendation
            invalid_recs_ = [rec for rec in invalid_recs if len(rec) < 50]
            valid_recs_ = [rec for rec in valid_recs if len(rec) < 50]
            
            invalid_recs_string = ":dislike, ".join(invalid_recs_) + ":dislike. "
            valid_recs_string = ":like, ".join(valid_recs_) + ":like. "
            
            next_input = (
                "The user would not enjoy the following recommendations: "
                + invalid_recs_string
                + "\nThe user would enjoy these recommendations: "
                + valid_recs_string
            )

        else:
            # Handles the case where precision is 100%
            next_input = "The user is likely to enjoy all of these recommendations. "

        if retry:
            # Should always be true up until the final prompt, where all recommendations will be consolidated into a final list
            next_input += (
                f"\nRecommend {self.k} additional movies released before "
                + str(self.knowledge_cutoff)
                + " that the user may enjoy, based on what you have learned about their preferences."
                + (f"\nTry to recommend movies that are less popular." if not self.allow_popular_movies else "")
                + "\nDo not reuse any recommendation that you see here."
                + "\nSort your recommendations by how confident you are that the user will like a movie."
            )

        else:
            next_input += self._build_final_prompt()

        next_input += (
            "\nProvide a short explanation for each recommendation."
            if self.allow_explanations
            else "\nDo not provide explanations for each recommendation."
        )

        return next_input

    def _build_final_prompt(self) -> str:
        final_input = (
            "Based on what you have learned about the user's preferences,"
            + f" recommend a final list of {self.final_k} additional movies released before {self.knowledge_cutoff} that the user might enjoy."
            + f"\nThis list should summarize the user's preferences, and be sorted by how confident you are that the user will like a movie."
        )
        return final_input
