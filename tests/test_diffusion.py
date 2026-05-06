import torch

from diffusion import generate_diffusion_labels


def test_generate_diffusion_labels_shape():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    y = generate_diffusion_labels(
        edge_index=edge_index,
        num_nodes=3,
        method="ic",
        params={"p": 0.5},
        mc_runs=4,
        max_steps=4,
        seed=7,
        n_jobs=1,
        use_cache=False,
    )
    assert tuple(y.shape) == (3, 1)
