import torch
from torch import nn
from collections import defaultdict
import torch
from torch import nn
import copy 

def build_model(dim, embedding_dim=3, init_stddev=1.,
                device="cuda", random_state=0):
    
    torch.random.manual_seed(random_state)
    # Embeddings for users
    U = torch.distributions.Normal(loc=0, scale=init_stddev).sample(
        (dim[0], embedding_dim)
        ).to(device=device)
    U.requires_grad = True
    
    # Embeddings for movies
    V = torch.distributions.Normal(loc=0, scale=init_stddev).sample(
        (dim[1], embedding_dim)
        ).to(device=device)
    V.requires_grad=True
    
    return NMFModel(U, V)


class NMFModel(nn.Module):
    def __init__(self, U, V):
        super().__init__()
        self._history = defaultdict(list)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        self._embedding_dim = U.shape[1]
        
    @property
    def ratings(self):
        return (self.U.detach() @ self. V.T.detach()).clone()
    
    def fit(self, train, test, n_iter, optimizer, loss, restore_best=False):
        if restore_best:
            best_param = copy.deepcopy(self.state_dict())
            best_loss = float("inf")

        for i in range(n_iter):
            optimizer.zero_grad()
            
            train_loss = 0
            test_loss = 0
            
            with torch.no_grad():
                self.U.abs_()
                self.V.abs_()
            
            for func in loss:
                train_loss += func(self.U,
                                   self.V,
                                   train)
                
                with torch.inference_mode():
                    test_loss += func(self.U,
                                      self.V,
                                      test)
           
            self._history["train"].append(train_loss.item())
            self._history["test"].append(test_loss.item())

            if restore_best and (test_loss < best_loss):
                best_param = copy.deepcopy(self.state_dict())
                best_loss = test_loss
            
            train_loss.backward()
            optimizer.step()
        
            print("Iteration: %i\t Train loss: %.3f\t Val loss: %.3f" % (i, train_loss, test_loss), end="\r")
        
        if restore_best:
            print("Restoring best weights with Val loss: %.3f" % (best_loss), end="\r")
            self.load_state_dict(best_param)
        
        with torch.no_grad():
            self.U.abs_()
            self.V.abs_()
        
        return self._history
    
    def predict(self, sparse_ratings, measure="dot"):
        # This is exactly the same as just taking the dot product of each query with each item
        with torch.inference_mode():
            gathered_user = torch.gather(
                self.U, 0, sparse_ratings.indices()[0].expand(self._embedding_dim, -1).T
                )
            gathered_item = torch.gather(
                self.V, 0, sparse_ratings.indices()[1].expand(self._embedding_dim, -1).T
                )
            
            if measure == "cosine":
                norm = torch.linalg.norm(gathered_user, dim=-1) * \
                    torch.linalg.norm(gathered_item, dim=-1)
            else:
                norm = 1
            
            predictions = torch.sum(gathered_user * gathered_item, dim=1) / norm
            
            # Faster to copy everything to cpu and then call item()
            combined = zip(sparse_ratings.indices()[0].cpu(),
                           sparse_ratings.indices()[1].cpu(),
                           predictions.cpu(), 
                           sparse_ratings.values().cpu()
                           )
            
            # Formatting the ratings into (user_id, item_id, estimated rating, true rating)
            user_pred = [(user.item(), item.item(), est.item(), true.item()) for user, item, est, true in combined]
        return user_pred
    
    def get_top_k(self, q, mode, k, measure, exclude_items=None):
        with torch.inference_mode():
            iid =  torch.arange(self.V.shape[0])

            if mode == "user":
                query = self.U[q]
            elif mode == "item":
                query = self.V[q]
            else:
                raise ValueError("Unexpected argument for 'mode.' Possible alternatives are: ('user', 'item')")
            
            if measure == "cosine":
                norm = torch.linalg.norm(query, dim=-1) * torch.linalg.norm(self.V, dim=-1)
            else:
                norm = 1
            
            sim = (query @ self.V.T) / norm

            if exclude_items:
                mask = torch.ones(self.V.shape[0], dtype=bool)
                mask[exclude_items] = False

                sim = sim[mask]
                iid = iid[mask]

            ratings = [(item.item(), score.item()) for item, score in zip(iid, sim)]

        return sorted(ratings, key=lambda x: x[1], reverse=True)[:k] 