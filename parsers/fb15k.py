from globals import READ_DIR, WRITE_DIR

import torch
from torch_geometric.data import Data
from tqdm import tqdm

IN_DIR = f'{READ_DIR}/FB15k-237/'
OUT_DIR = f'{WRITE_DIR}/FB15k/'
TO_PARSE = ['train.txt','valid.txt','test.txt']

rel_types = dict()
ent_types = dict()

def get_id(s: str, db: dict) -> int:
    if (idx := db.get(s)) is None:
        idx = len(db)
        db[s] = idx
    return idx

def parse_one(fname):
    f = open(fname, 'r')
    line = f.readline()
    ei, et = [[],[]], []
    prog = tqdm()

    while line:
        src,rel,dst = line.split('\t')
        src = get_id(src, ent_types)
        rel = get_id(rel, ent_types)
        dst = get_id(dst, ent_types)

        ei[0].append(src)
        ei[1].append(dst)
        et.append(rel)

        line = f.readline()
        prog.update()

    prog.close()
    return Data(
        edge_index = torch.tensor(ei, dtype=torch.long),
        edge_attr  = torch.tensor(et, dtype=torch.long)
    )

if __name__ == '__main__':
    for fname in TO_PARSE:
        in_fname = f'{IN_DIR}/{fname}'
        out_fname = f'{OUT_DIR}/{fname}'.replace('.txt', '.pt')

        data = parse_one(in_fname)
        torch.save(data, out_fname)