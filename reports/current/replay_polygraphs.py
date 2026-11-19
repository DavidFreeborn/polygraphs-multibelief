"""Replay saved evidence through the actual PolyGraphs operations."""

import argparse
import gzip
import json
from pathlib import Path

import dgl
import numpy as np
import torch

from polygraphs import hyperparameters as hp
from polygraphs.beliefs.cases import binary_node
from polygraphs.ops import ReportedMultiBeliefOp, BatchedBayesianReliabilityOp

HERE = Path(__file__).resolve().parent


def specifications(model):
    nodes = [binary_node('H')]
    if model:
        nodes.append(binary_node('S'))
    if model == 3:
        nodes = [binary_node('H'), binary_node('S', ['X', 'Y'], [[.9,.1],[.1,.9]]), binary_node('X'), binary_node('Y')]
    if model == 0:
        observation = {'parents':['H'], 'probability':[.45,.55]}
    elif model == 1:
        observation = {'parents':['H','S'], 'probability':[[.45,.45],[.45,.55]]}
    else:
        observation = {'parents':['H','S'], 'probability':[[.55,.45],[.45,.55]]}
    return nodes, observation


def replay(path):
    run = json.loads(path.read_text())
    source, recipient = np.array(run['edges']).T
    graph = dgl.graph((source,recipient), num_nodes=60)
    p = hp.PolyGraphHyperParameters()
    p.trials, p.epsilon = 1, .05
    p.multibelief.backend = 'numpy'
    p.multibelief.stop_on_consensus = False
    p.simulation.steps = run['rounds']
    p.unreliablenodes = run['bad']
    trust = run['model'] == 4
    if trust:
        p.bayesian_reliability.prior = run['trust_prior']
        p.bayesian_reliability.unreliable_mode = 'opposite' if run['environment'] == 2 else 'uninformative'
        model = BatchedBayesianReliabilityOp(graph,p)
        model.prior_h = np.array(run['initial'])
        model.log_h = model.prior_h.copy()
    else:
        nodes,observation = specifications(run['model'])
        p.multibelief.case = 'custom'
        p.multibelief.target = 'H'
        p.multibelief.prior_mode = 'fixed'
        p.multibelief.model = {'nodes':nodes}
        p.multibelief.observation = observation
        p.multibelief.true_probability = .55
        p.multibelief.source_mode = ['honest','uninformative','opposite'][run['environment']]
        model = ReportedMultiBeliefOp(graph,p)
        # The worker enumerates H as the fastest axis; PolyGraphs uses C order.
        bits=len(nodes)
        mapping=np.array([sum(int(model.assignments[s,k])<<k for k in range(bits)) for s in range(1<<bits)])
        model.initial_log_joint=np.array(run['initial'])[:,mapping]
        model.log_joint=model.initial_log_joint.copy()
    model._publish(graph)
    raw=np.frombuffer(gzip.decompress((HERE/'data/raw'/f'{path.stem}.gz').read_bytes()),dtype=np.uint8).reshape(-1,60)
    for codes in raw:
        active=codes!=255
        def experiment(g):
            g.ndata['private_payoffs']=torch.tensor(np.stack((np.where(active,codes&15,0),active),1),dtype=torch.float64)
            g.ndata['payoffs']=torch.tensor(np.stack((np.where(active,codes>>4,0),active),1),dtype=torch.float64)
        model.experiment=experiment
        model(graph)
    expected=np.array(run['final'])
    actual=model.log_h if trust else model.log_joint[:,np.argsort(mapping)]
    error=float(np.max(np.abs(actual-expected)))
    assert error < 1e-8,(path.name,error)
    return error


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    paths=sorted((HERE/'data/runs').glob('*.json'))
    if not args.all:paths=[p for p in paths if p.stem.endswith('-1729')]
    maximum=max(replay(p) for p in paths)
    print(json.dumps({'runs':len(paths),'max_log_posterior_error':maximum},indent=2))


if __name__=='__main__':main()
