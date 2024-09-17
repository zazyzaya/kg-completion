import glob
import datetime as dt

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from globals import READ_DIR, WRITE_DIR

IN_DIR = f'{READ_DIR}/ICEWS/'
OUT_DIR = f'{WRITE_DIR}/ICEWS/'

TS=1; SRC=2; SRC_SECTORS=3; SRC_COUNTRY=4
EVENT=5; DST=8; DST_SECTORS=9; DST_COUNTRY=10

rel_types = dict()
ent_types = dict()

def get_id(s: str, db: dict) -> int:
    if (idx := db.get(s)) is None:
        idx = len(db)
        db[s] = idx
    return idx

def get_relevant(tokens):
    src = get_id(tokens[SRC], ent_types)
    dst = get_id(tokens[DST], ent_types)
    rel = get_id(tokens[EVENT], rel_types)

    y,m,d = tokens[TS].split('-')
    ts = dt.datetime(int(y), int(m), int(d)).timestamp()

    # I think we can extract a lot of implicit edges from the sectors and country fields
    # but maybe that's a task for another time.

    return src,dst,rel,ts

def parse_one(fname):
    f = open(fname, 'r')
    line = f.readline() # Skip header
    line = f.readline()
    ei, et, timestamps = [[],[]], [], []
    prog = tqdm()

    while line:
        tokens = line.split('\t')
        src,dst,rel,ts = get_relevant(tokens)

        ei[0].append(src)
        ei[1].append(dst)
        et.append(rel)
        timestamps.append(ts)

        line = f.readline()
        prog.update()

    prog.close()
    return Data(
        edge_index = torch.tensor(ei, dtype=torch.long),
        edge_attr  = torch.tensor(et, dtype=torch.long),
        ts = torch.tensor(timestamps)
    )

if __name__ == '__main__':
    for fname in glob.glob(f'{IN_DIR}/*.tab'):
        year = fname.split('.')[-3]
        out_fname = f'{OUT_DIR}/{year}.pt'

        data = parse_one(fname)
        torch.save(data, out_fname)