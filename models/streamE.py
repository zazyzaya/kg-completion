import torch
from torch import nn

'''
Adapted from https://github.com/zjs123/StreamE/blob/main/model/Model.py
'''

class Updater(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.z = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )

        self.c = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Tanh()
        )

    def forward(self, h, v):
        hv = torch.cat([h,v], dim=-1)
        z = self.z(hv)
        c = self.c(hv)

        h_t = (1-z)*h + z*c
        return h_t

class StreamE(nn.Module):
    def __init__(self, emb_dim, num_entities, num_relations, n_time_params=10):
        super().__init__()
        self.emb_dim = emb_dim

        self.relation_embeddings = nn.Embedding(num_relations, emb_dim)
        self.entity_embeddings = nn.Embedding(num_entities, emb_dim, padding_idx=num_entities)

        # Calculate (new) node embeddings, h
        self.prop_influence_net = nn.Linear(emb_dim*3, emb_dim)
        self.update_gate = Updater(emb_dim)

        # Calculate temporal periodicity, e (unused in StreamE repo)
        self.periods = torch.tensor([ [(i+1)**2] for i in range(n_time_params)], dtype=torch.float)
        self.time_param = nn.Parameter(torch.empty((n_time_params, emb_dim), requires_grad=True))
        self.time_net = nn.Linear(emb_dim*2, emb_dim)
        nn.init.xavier_normal_(self.time_param)

        # Calculate semantic relevance, g (also unused in StreamE repo)
        self.semantic_rel_net = nn.Linear(emb_dim)


    def get_embedding(self, g, subjects, relations, objects, t):
        h_t = torch.zeros(subjects.size(0), self.hidden_dim)

        noninductive = g.first_seen[subjects] < t
        inductive = ~noninductive
        has_embedding = subjects[noninductive]
        h_t[noninductive] = self.entity_embeddings[has_embedding]

        # No need to update anything
        if inductive.sum() == 0:
            return h_t

        # Otherwise, find any subject nodes that were heads of < *, rel, obj >
        similar_subjects, st_en, _ = g.edge_csr.get_subjects(
            objects[inductive],
            t=t[inductive],
            rel=relations[inductive]
        )
        similar_h = self.entity_embeddings[similar_subjects]

        nids = inductive.nonzero().squeeze(-1)
        for i in range(nids.size(0)):
            nid = nids[i]
            h_sim = similar_h[st_en[i]:st_en[i+1]]

            # Just use mean of all subject nodes for this relation if no matching
            # object node is available.
            if h_sim.size(0) == 0:
                h_sim = g.edge_csr.get_all_subjects(relations[inductive[i]])

            h_t[nid] = h_sim.mean(dim=0)

        return h_t

    def update_embeddings(self, g, new_nodes, t):
        one_hop, st_en, rel1 = g.edge_csr.get_neighbors(new_nodes, t, undirected=True)

        d = torch.zeros(one_hop.size(0), self.emb_dim)
        for i in range(one_hop.size(0)):
            st,en = st_en[i], st_en[i+1]
            d[i] =

        one_hop_uq, oh_uq = one_hop.unique(return_inverse=True)
        two_hop, st_en2, rel2 = g.edge_csr.get_neighbors(one_hop, t, undirected=True)

