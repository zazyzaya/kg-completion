import torch

class CSR():
    def __init__(self, ei, ef, et, reverse=True):
        self.index = []
        self.objs = []
        self.times = []
        self.rels = []
        self.n = 0

        self.__build(ei,ef,et)

        if reverse:
            self.reverse = CSR(ei[[1,0]], ef, et, reverse=False)

    def __build(self, ei,ef,et):
        self.n = ei.max()+1

        objects = [[]] * self.n
        times = [[]] * self.n
        rels = [[]] * self.n

        for i in range(ei.size(0)):
            s,o = ei[:,i]
            s = s.item()

            objects[s].append(o.item())
            times[s].append(et[i].item())
            rels[s].append(ef[i].item())

        index = [0]
        for obj in objects:
            index.append(index[-1] + len(obj))

        self.index = torch.tensor(index)
        self.objs = torch.tensor(sum(objects, []))
        self.times = torch.tensor(sum(times, []))
        self.rels = torch.tensor(sum(rels, []))

        # Regularize
        self.times = self.times - self.times.min()

    def get_one(self, subject, target_rel=None, time=None):
        st,en = self.index[subject], self.index[subject+1]

        obj = self.objs[st:en]
        ts = self.times[st:en]
        rels = self.rels[st:en]

        mask = torch.ones(obj.size(0), dtype=bool)
        if time is not None:
            mask = mask * (ts <= time)
        if target_rel is not None:
            mask = mask * (rels == target_rel)

        return obj[mask], ts[mask], rels[mask]

    def get_neighbors(self, subjects, target_rels=None, times=None, undirected=False):
        ret = []
        rels = []
        index = [0]

        if target_rels is None:
            target_rels = [None] * subjects.size(0)
        if times is None:
            times = [None] * subjects.size(0)

        for i,subject in enumerate(subjects):
            obj, ts, rel = self.get_one(subject, target_rels[i], times[i])

            if undirected:
                obj_, ts_, rels_ = self.reverse.get_one(subject, target_rels[i], times[i])
                obj = torch.cat([obj, obj_])
                ts = torch.cat([ts, ts_])
                rel = torch.cat([rel, rels_])

            ret.append(obj)
            rels.append(rel)
            index.append(index[-1] + obj.size(0))

        ret,idx,rels = torch.cat(ret), torch.tensor(index), torch.cat(rels)
        return ret,idx,rels

    def get_subjects(self, objects, target_rels=None, times=None):
        return self.reverse.get_neighbors(objects, target_rels, times)

    def get_all_subjects(self, target_rel, time):
        neighs, _, _ = self.reverse.get_neighbors(
            torch.arange(self.n),
            torch.tensor([[target_rel.item()]]).repeat(self.n, 1),
            torch.tensor([[time.item()]]).repeat(self.n, 1)
        )

        ret = set([n.item() for n in neighs])
        return torch.tensor(list(ret))
