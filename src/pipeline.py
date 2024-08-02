import os
import asyncio
import json
import time
from typing import Iterable

from src.recommender import RecParam, GPTRec, DummyRec, NMFRec, LlamaRec
from src.prompt import Prompter
from src.evaluation import Evaluator, construct_user, construct_set, User
from src.dictionary import ItemDictionary


def init_result_files(param: RecParam):
    """Initializes result files based on path given in param set

    Args:
        param (RecParam): Collection of parameters
    """

    model_name = param.model
    if "/" in model_name:
        model_name.replace("/", "//")

    new_result_path = os.path.join(
        param.result_path,
        f"k={param.k}_p={param.num_prompts}_prompt={param.prompt_style}_level={param.embedding_level}_model={model_name}_aux={param.use_aux}_evalstyle={param.evaluation_style}_select={param.selection_style}_explain={param.allow_explanations}",
        f"result_{int(time.time())}",
    )

    os.makedirs(new_result_path)
    # Store specific run result path for saving later
    param.result_path = new_result_path

    with open(os.path.join(new_result_path, "metrics.csv"), "w+") as f:
        param_fields = [
            "uid",
            "random_state",
            "k",
            "final_k",
            "allow_explanations",
            "num_prompts",
            "temperature",
            "selection_style",
            "prompt_style",
            "embedding_level",
            "title_threshold",
            "min_sim_quantile",
            "model",
            "usage",
            "unmatched_count",
        ]
        size_fields = [
            "like_example_size",
            "like_feedback_size",
            "like_eval_size",
            "dislike_example_size",
            "dislike_feedback_size",
            "dislike_eval_size",
        ]

        recall_fields = ["recall_" + str(i) for i in range(param.num_prompts - 1)] + [
            "recall_final"
        ]
        precision_fields = [
            "precision_" + str(i) for i in range(param.num_prompts - 1)
        ] + ["precision_final"]
        ncdg_fields = ["ndcg_" + str(i) for i in range(param.num_prompts - 1)] + [
            "ndcg_final"
        ]
        ils_fields = ["ils_" + str(i) for i in range(param.num_prompts - 1)] + [
            "ils_final"
        ]
        ap_fields = ["ap_" + str(i) for i in range(param.num_prompts - 1)] + [
            "ap_final"
        ]
        dq_fields = ["dq_" + str(i) for i in range(param.num_prompts - 1)] + [
            "dq_final"
        ]
        combined_fields = (
            param_fields
            + size_fields
            + recall_fields
            + precision_fields
            + ncdg_fields
            + ils_fields
            + ap_fields
            + dq_fields
        )

        row = ",".join(i for i in combined_fields) + "\n"
        f.write(row)


async def write_results(
    param: RecParam,
    user: User,
    evaluator: Evaluator,
    recommender: GPTRec,
    all_predicted_titles_user: Iterable[str],
):
    # Write the metrics with specific formatting
    with open(os.path.join(param.result_path, "metrics.csv"), "a") as f:
        param_info = [
            user.uid,
            param.random_state,
            param.k,
            param.final_k,
            param.allow_explanations,
            param.num_prompts,
            param.temperature,
            param.selection_style,  # ONE OF KMEDOIDS/RANDOM
            param.prompt_style,
            param.embedding_level,
            param.title_threshold,
            param.min_sim_quantile,
            param.model,
            recommender.usage,
            evaluator.unmatched_count,
        ]
        sizes = user.get_sizes()
        recalls = evaluator.recall_at_p
        precisions = evaluator.precision_at_p
        ndcgs = evaluator.ndcg_at_p
        ils = evaluator.ils_at_p
        aps = evaluator.ap_at_p
        dqs = evaluator.delinquency_at_p
        combined = param_info + sizes + recalls + precisions + ndcgs + ils + aps + dqs

        row = ",".join(str(i) for i in combined) + "\n"
        f.write(row)

    # Record the actual conversation (enables a checkpoint if necessary)
    with open(os.path.join(param.result_path, "conversation.jsonl"), "a") as f:
        f.write(json.dumps(recommender.get_messages(user.uid)) + "\n")

    # Record all predictions generated
    with open(os.path.join(param.result_path, "predicted_titles.jsonl"), "a") as f:
        f.write(json.dumps(all_predicted_titles_user) + "\n")

    with open(os.path.join(param.result_path, "unmatched_titles.txt"), "a") as f:
        for title in evaluator.unmatched_titles:
            try:
                f.write(title + "\n")
            except UnicodeEncodeError:
                pass


async def single_conversation(
    uid: int,
    user_split: dict[list, list],
    item_dict: ItemDictionary,
    prompter: Prompter,
    param: RecParam,
):
    ##################################################
    # SETUP EACH ENTITY IN THE PIPELINE
    ##################################################
    user = construct_user(uid, user_split, item_dict)

    evaluator = Evaluator()

    like_feedback_set = construct_set(user, "like", "feedback", item_dict)
    dislike_feedback_set = construct_set(user, "dislike", "feedback", item_dict)

    ########################################################
    # RECOMMENDER TYPE
    ########################################################
    if param.model == "dummy":
        recommender = DummyRec(title_list=item_dict.get_all_titles(), random_state=None)

    elif "nmf" in param.model:
        recommender = NMFRec()

    elif "llama" in param.model:
        recommender = LlamaRec(model=param.model, temperature=param.temperature)

    else:
        recommender = GPTRec(model=param.model, temperature=param.temperature)

    # NOTE: this needs to be moved outside. It doesn't make sense to do this here

    ###################################################################################
    # INITIAL MESSAGE (DETERMINED BY PROMPTING STYLE: COT/FEW-SHOT/ZERO-SHOT)
    ###################################################################################

    if param.prompt_style == "zero":
        init_prompt = prompter.build_zeroshot_prompt(
            like_titles=user.titles["like"]["example"],
            dislike_titles=user.titles["dislike"]["example"],
        )

    elif param.prompt_style == "few":
        init_prompt = prompter.build_fewshot_prompt(
            like_titles=user.titles["like"]["example"],
            dislike_titles=user.titles["dislike"]["example"],
            provide_reasoning=False,
            random_state=param.random_state,
            item_dict=item_dict,
        )

    elif param.prompt_style == "cot":
        init_prompt = prompter.build_fewshot_prompt(
            like_titles=user.titles["like"]["example"],
            dislike_titles=user.titles["dislike"]["example"],
            provide_reasoning=True,
            random_state=param.random_state,
            item_dict=item_dict,
        )

    recommender.add_message(init_prompt)

    ##################################################
    # BEGIN PROMPT LOOP
    ##################################################
    for p in range(param.num_prompts - 1):
        # For NMF variant
        if "user" in param.model:
            pred_iids = recommender.recommend_next_user(uid, param.k)
            pred_titles = item_dict.get_title_from_iid(pred_iids)
        elif "item" in param.model:
            pred_iids = recommender.recommend_next_item(user.iids["like"]["feedback"])
            pred_titles = item_dict.get_title_from_iid(pred_iids)
        elif "dummy" in param.model:
            pred_titles = recommender.recommend_next(param.k)
        else:
            pred_titles = await recommender.recommend_next()

        pred_embeddings = item_dict.get_embedding_from_title(
            titles=pred_titles, title_threshold=param.title_threshold
        )  # k embeddings

        # Compute metrics (recall/precision/ndcg) and determine wrong recommendations
        invalid_recs, valid_recs = evaluator.evaluate(
            like_feedback_set,
            dislike_feedback_set,
            pred_embeddings,
            pred_titles,
            eval_style=param.evaluation_style,
        )

        if p < (param.num_prompts - 2):
            retry = True
        else:
            retry = False

        # Set up recommender for next prompt
        next_input = prompter.build_next_prompt(invalid_recs, valid_recs, retry=retry)
        recommender.add_message(next_input)

    ##################################################
    # REPEAT EVALUATION ON FINAL PROMPT
    ##################################################

    like_eval_set = construct_set(user, "like", "eval", item_dict)
    dislike_eval_set = construct_set(user, "dislike", "eval", item_dict)

    if "user" in param.model:
        pred_iids = recommender.recommend_next_user(uid, param.final_k)
        pred_titles = item_dict.get_title_from_iid(pred_iids)
    elif "item" in param.model:
        pred_iids = recommender.recommend_next_item(
            user.iids["like"]["eval"], param.final_k
        )
        pred_titles = item_dict.get_title_from_iid(pred_iids)
    elif "dummy" in param.model:
        pred_titles = recommender.recommend_next(param.final_k)
    else:
        pred_titles = await recommender.recommend_next()

    pred_embeddings = item_dict.get_embedding_from_title(
        titles=pred_titles, title_threshold=param.title_threshold
    )  # k embeddings

    # Compute metrics (recall/precision/ndcg) and determine wrong recommendations
    evaluator.evaluate(
        like_eval_set,
        dislike_eval_set,
        pred_embeddings,
        pred_titles,
        eval_style=param.evaluation_style,
    )

    all_predicted_titles_user = {uid: evaluator.all_predicted_titles}

    # Don't write to results if result_path not set
    if param.result_path:
        await write_results(
            param, user, evaluator, recommender, all_predicted_titles_user
        )


async def repeat_replicates(
    uid: int,
    user_split: dict[dict[dict[Iterable]]],
    item_dict: ItemDictionary,
    prompter: Prompter,
    param: RecParam,
):
    group = asyncio.gather(
        *[
            single_conversation(uid, user_split, item_dict, prompter, param)
            for _ in range(param.num_replicates)
        ]
    )
    await group


def batched_users(
    user_items: Iterable[dict[Iterable]], batch_size: int
) -> Iterable[dict[Iterable]]:
    """
    Used to avoid possible token/rate limits by starting N number of async prompt loops at once
    """
    start = 0
    stop = start + batch_size

    while start < len(user_items):
        yield user_items[start:stop]

        start = stop
        stop = start + batch_size


async def recommend_per_user(
    user_splits: Iterable[dict[dict[dict[Iterable]]]],
    item_dict: ItemDictionary,
    prompter: Prompter,
    param: RecParam,
    batch_size: int = 5,
):
    # If this is None, don't save results
    if param.result_path:
        init_result_files(param)

    for user_batch in batched_users(list(user_splits.items()), batch_size):
        group = asyncio.gather(
            *[
                repeat_replicates(uid, splits, item_dict, prompter, param)
                for uid, splits in user_batch
            ]
        )

        await group

    param.save()
