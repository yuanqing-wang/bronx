import dgl
import torch

def new_edges(
        g: dgl.DGLGraph,
        percentage: float,
):
    """Add new edges to a graph.
    
    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.

    percentage : float
        The percentage of edges to add.

    Returns
    -------
    dgl.DGLGraph
        The graph with new edges.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))
    >>> new_edges(g, 0.5)
    DGLGraph(num_nodes=3, num_edges=6,
             ndata_schemes={}
             edata_schemes={})

    """
    num_edges = int(g.number_of_edges() * percentage)
    nodes = g.nodes()
    src = torch.randint(0, nodes, (num_edges,))
    dst = torch.randint(0, nodes, (num_edges,))
    g.add_edges(src, dst)
    return g

def delete_edges(
        g: dgl.DGLGraph,
        percentage: float,
):
    """Delete edges from a graph.
    
    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.

    percentage : float
        The percentage of edges to delete.

    Returns
    -------
    dgl.DGLGraph
        The graph with deleted edges.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))
    >>> delete_edges(g, 0.5)
    DGLGraph(num_nodes=3, num_edges=1,
             ndata_schemes={}
             edata_schemes={})

    """
    num_edges = int(g.number_of_edges() * percentage)
    edges = torch.randint(0, g.number_of_edges(), (num_edges,))
    g.remove_edges(edges)
    return g
