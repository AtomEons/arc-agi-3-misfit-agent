"""Random control agent — pure random selection within legal options."""
import os, random
from cg.api import Observation, to_observation_class

def read_deck_csv():
    fp = "deck.csv"
    if not os.path.exists(fp):
        fp = "/kaggle_simulations/agent/" + fp
    with open(fp) as f:
        return [int(r.strip()) for r in f.read().split() if r.strip()][:60]

def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    n_opts = len(obs.select.option)
    take = max(obs.select.minCount, 1) if obs.select.maxCount >= 1 else obs.select.minCount
    take = min(take, obs.select.maxCount, n_opts)
    return random.sample(range(n_opts), take)
