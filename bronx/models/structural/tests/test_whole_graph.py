
def test_whole_graph():
    import torch
    import dgl
    import bronx
    from bronx.models.zoo.dgl import GCN
    from bronx.models.structural.model import StructuralModel
    from bronx.models.head.node_classification import NodeClassificationPyroHead
    g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    h = torch.randn(3, 10)
    y = torch.tensor([0, 1, 2])
    model = StructuralModel(
        head=NodeClassificationPyroHead(),
        layer=GCN,
        in_features=10,
        out_features=3,
        hidden_features=15,
        depth=2,
    )
    model.training_step((g, h, y), 0)

