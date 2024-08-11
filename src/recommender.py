import backoff
import os
import re
import openai
from typing import Optional, Union, Iterable
import dataclasses
import json
import numpy as np
import torch
import pickle

import replicate
import replicate.exceptions
import httpcore


@dataclasses.dataclass
class RecParam:
    result_path: str
    k: int
    final_k: int
    allow_explanations: bool
    num_prompts: int
    num_examples: Union[float, int]
    eval_size: float
    temperature: float
    num_replicates: int
    selection_style: str  # Either stratified or kmedoids
    evaluation_style: str
    prompt_style: str
    embedding_level: int
    title_threshold: float
    min_sim_quantile: float
    random_state: int
    model: str
    use_aux: bool
    allow_popular_movies: bool

    def save(self):
        with open(os.path.join(self.result_path, "parameters.json"), "w") as f:
            f.write(json.dumps(dataclasses.asdict(self)))
            
    @staticmethod
    def from_json(path: str): 
        with open(path, "r") as f:
            param = RecParam(**json.load(f))
        return param

item_filter_pat = re.compile(r"(?:\d\.\s+)(\w.+\(\d+\))")

def extract_gpt_rec(response: str) -> list[str]:
    titles = item_filter_pat.findall(response)
    if len(titles) < 5:
        titles = item_filter_pat.findall(response)
    return titles

def build_llama_prompt(messages: list[dict]) -> str:
    assert len(messages) > 1, "build_llama_prompt called prematurely"
    
    prompt = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"

    prompt = prompt.format(system_prompt=messages[0]["content"], prompt=messages[1]["content"])

    for msg in messages[2:]:
        role = msg["role"]
        if role == "assistant":
            prompt += "\n{}".format(msg["content"])
        elif role == "user":
            prompt += "\n[INST]{} [/INST]".format(msg["content"])
            
    return prompt

class RecommenderCore:
    def set_k(self, k: int) -> None:
        self._k = k
        
    def restart(self) -> None:
        """Returns messages to only context"""
        self._messages = [self._messages[0]]

    def add_message(self, input_string: str) -> None:
        self._messages += [{"role": "user", "content": input_string}]

    def get_messages(self, uid: int) -> list[dict[str]]:
        conversation_with_id = {"user": uid, "messages": self._messages}
        return conversation_with_id 
    
class LlamaRec(RecommenderCore):
    def __init__(self, model: str, temperature: float = 0.0, context: Optional[str] = None):
        super().__init__()
        self._messages = [
            {
                "role": "system",
                "content": context or "You are a movie recommender system.",
            },
        ]
        
        self._temperature = temperature
        self._model = model
        self.usage = 0
        
    @backoff.on_exception(
    backoff.constant,
    httpcore.ConnectTimeout,
    interval=30,
    max_tries=7,
    jitter=None,
    ) 
    @backoff.on_exception(
    backoff.constant,
    httpcore.ReadTimeout,
    interval=30,
    max_tries=5,
    jitter=None,
    ) 
    @backoff.on_exception(
    backoff.constant,
    replicate.exceptions.ModelError,
    interval=30,
    max_tries=5,
    jitter=None,
    ) 
    async def recommend_next(self, return_content=False) -> list[str]:
        prompt = build_llama_prompt(self._messages)
        
        request = {
            "prompt": prompt,
            "temperature": self._temperature,
            "max_new_tokens": 1500,
        }
        
        response = await replicate.async_run(
            ref=self._model,
            input=request
        )
        
        # Need to join output because it is given as a stream
        response = "".join(response)
        
        new_message = {"role": "assistant", "content": response}
       
        self._messages.append(new_message)  # For re-prompting purposes

        # Extract content
        recommendations = extract_gpt_rec(response)

        if return_content:
            return recommendations, response
        else:
            return recommendations
        
class GPTRec(RecommenderCore):
    def __init__(self, model: str, temperature: float = 0.2, context: Optional[str] = None):
        super().__init__()
        self._messages = [
            {
                "role": "system",
                "content": context or "You are a movie recommender system.",
            },
        ]

        # self._knowledge_cutoff = knowledge_cutoff
        self._temperature = temperature
        self._model = model
        self.usage = 0
    
    @backoff.on_exception(
        backoff.expo,
        openai.APIError,
        factor=5,
        max_value=30,
        max_tries=10,
        jitter=None,
    )
    @backoff.on_exception(
        backoff.expo,
        openai.APITimeoutError,
        factor=5,
        max_value=30,
        max_tries=10,
        jitter=None,
        # logger=logger,
    )
    @backoff.on_exception(
        backoff.constant,
        openai.RateLimitError,
        interval=25,
        max_tries=5,
        jitter=None,
    )
    @backoff.on_exception(
        backoff.constant,
        openai.InternalServerError,
        interval=30,
        max_tries=5,
        jitter=None,
    )
    async def recommend_next(self, return_content=False) -> list[str]:
        response = await openai.ChatCompletion.acreate(
            #model="gpt-3.5-turbo",
            model=self._model,
            messages=self._messages,
            temperature=self._temperature,
        )
        # Save current usage info
        self.usage = response["usage"]["total_tokens"]

        # Grab message from response
        new_message = response["choices"][0]["message"]
        self._messages.append(new_message)  # For re-prompting purposes

        # Extract content
        content = new_message["content"]
        recommendations = extract_gpt_rec(content)

        # embeddings = np.array(list(map(
        #    lambda x: get_embedding(x, engine=embedding_model), recommendations
        #    )))
        if return_content:
            return recommendations, content
        else:
            return recommendations

class DummyRec(RecommenderCore):
    """
    Implements the random recommender
    """
    def __init__(self, title_list: Iterable[str], random_state: int):
        super().__init__()
        self._messages = [
            {"role": "system", "content": "This is a dummy."},
        ]

        self._title_list = title_list
        self._rng = np.random.default_rng(random_state)
        self.usage = 0

    def recommend_next(self, k: int, return_content=False) -> list[str]:
        # Extract content
        recommendations = self._rng.choice(
            self._title_list, size=k, replace=False
        ).tolist()

        for title in recommendations:
            self._title_list.remove(title)

        self._messages += [{"role": "system", "content": ",".join(recommendations)}]

        if return_content:
            return recommendations, None
        else:
            return recommendations

    def get_messages(self, uid: int) -> list[dict[str]]:
        conversation_with_id = {"user": uid, "messages": self._messages}
        return conversation_with_id

from src.nmf import build_model, NMFModel

class NMFRec(RecommenderCore):
    def __init__(self):
        super().__init__()
        self._messages = [
            {"role": "system", "content": "This is NMF."},
        ]

        self._model = torch.load("./src/nmf_model.pt", map_location=torch.device("cpu"))
        self._model.eval()
        self.usage = 0

        with open("./src/nmf_id_map.pickle", "rb") as f:
            self._map = pickle.load(f)

    def recommend_next_user(self, uid: int, k :int) -> Iterable[int]:
        uid_enc = self._map["uid_to_enc"][uid]
        
        top_k = self._model.get_top_k(
            q=uid_enc, mode="user", k=k, measure="dot", exclude_items=[uid_enc]
        )

        enc_iids, _ = zip(*top_k)
        true_iids = [self._map["enc_to_iid"][i] for i in enc_iids]

        return true_iids
    
    def recommend_next_item(self, iids: Iterable[int], k: int) -> Iterable[int]:
        enc_iids = [self._map["iid_to_enc"][i] for i in iids]

        example_embeddings = self._model.V[enc_iids[0]].unsqueeze(0)
        for i in enc_iids[1:]:
            example_embeddings = torch.concat((example_embeddings, self._model.V[i].unsqueeze(0)), dim=0)
        
        pool = []
        for i in enc_iids:
            rec_items, _ = zip(*self._model.get_top_k(q=i, k=k, mode="item", measure="cosine", exclude_items=pool))
            pool += rec_items
            
        pool_embeddings = self._model.V[pool[0]].unsqueeze(0)

        for i in pool[1:]:
            pool_embeddings = torch.concat((pool_embeddings, self._model.V[i].unsqueeze(0)), dim=0)
            
        most_similar_ind = torch.argsort(torch.sum((example_embeddings @ pool_embeddings.T), axis=0))[-self._k:]
        
        true_iids = [self._map["enc_to_iid"][i.item()] for i in most_similar_ind]
        return true_iids