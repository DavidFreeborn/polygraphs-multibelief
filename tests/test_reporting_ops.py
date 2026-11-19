"""The source's own observation is distinct from its outgoing report."""

import dgl
import numpy as np
import torch

from polygraphs import hyperparameters as hp
from polygraphs.ops import ReportedMultiBeliefOp


def test_private_observation_is_not_replaced_by_own_report():
    p=hp.PolyGraphHyperParameters()
    p.epsilon=.05
    p.trials=10
    p.init.kind='constant'
    p.init.constant.value=.5
    p.multibelief.backend='numpy'
    p.multibelief.stop_on_consensus=False
    p.simulation.steps=1
    graph=dgl.graph(([0],[1]),num_nodes=2)
    model=ReportedMultiBeliefOp(graph,p)
    def experiment(g):
        g.ndata['private_payoffs']=torch.tensor([[8,10],[0,0]],dtype=torch.float64)
        g.ndata['payoffs']=torch.tensor([[2,10],[0,0]],dtype=torch.float64)
    model.experiment=experiment
    model(graph)
    odds=(.55/.45)**6
    np.testing.assert_allclose(graph.ndata['beliefs'].numpy(),[odds/(1+odds),1/(1+odds)],atol=1e-12)
