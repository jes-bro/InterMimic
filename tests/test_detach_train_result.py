"""detach_train_result (learning/common_agent.py): the per-minibatch result dict
must not keep autograd graphs alive.

Why: the distill agent computes the actor and critic losses every minibatch but
only puts the critic in `loss` after epoch 2300 (actor: 2900). A loss tensor
that never went through backward still owns its graph and the activations saved
for it; stored in self.train_result and appended to the epoch's train_info lists,
24 such graphs (6 mini-epochs x 4 minibatches) of a separate transformer critic
held ~9.5 GB through every rollout of job 17894921's first 2300 epochs, and the
caching allocator kept that high-water mark reserved until PhysX died on a
64 MB allocation at 44.2/44 GB.

common_agent.py imports rl_games/isaacgym (unavailable locally), so the test
extracts the function's source and execs it -- the code as shipped, without
the module's import graph (same pattern as test_exact_policy_kl.py).

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_detach_train_result.py -q
"""

import os
import re

import torch

LEARNING = os.path.join(os.path.dirname(__file__), "..", "isaacgym", "src", "intermimic", "learning")
AGENT_FILE = os.path.join(LEARNING, "common_agent.py")


def _load_detach_train_result():
    src = open(AGENT_FILE).read()
    m = re.search(r"(def detach_train_result\(result\):.*?\n    return result\n)", src, re.S)
    assert m, "detach_train_result not found in common_agent.py"
    ns = {"torch": torch}
    exec(m.group(1), ns)
    return ns["detach_train_result"]


def _unbackpropped_minibatch():
    """A 'critic' forward whose loss is computed but never backpropagated, the
    pre-epoch-2300 situation. Returns (result dict, the loss, a weak ref to the
    critic's input activation): Linear's backward saves its input, so after this
    function returns only the loss's graph keeps that activation alive -- the
    stand-in for the transformer critic's activations the leak retained."""
    import weakref

    critic = torch.nn.Linear(8, 1)
    x = torch.randn(4, 8)
    values = critic(x)
    c_loss = ((values - torch.zeros(4, 1)) ** 2).mean()
    result = {
        "entropy": torch.tensor(0.5),          # plain tensor, no graph
        "kl": torch.tensor(0.01),
        "last_lr": 2e-5,                       # non-tensors pass through
        "lr_mul": 1.0,
        "critic_loss": c_loss,                 # owns a graph
        "actor_clipped": torch.zeros(4).bool(),
    }
    return result, c_loss, weakref.ref(x)


def test_detaches_every_tensor_in_place_and_keeps_values():
    detach = _load_detach_train_result()
    result, _, _ = _unbackpropped_minibatch()
    assert result["critic_loss"].requires_grad and result["critic_loss"].grad_fn is not None
    before = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in result.items()}

    out = detach(result)

    assert out is result, "must detach in place (callers keep the same dict)"
    for k, v in result.items():
        if torch.is_tensor(v):
            assert not v.requires_grad and v.grad_fn is None, k
            assert torch.equal(v, before[k]), k      # values untouched
        else:
            assert v == before[k], k                  # non-tensors untouched


def test_detached_dict_releases_the_graph():
    """The graph (and the activations it saved) must become collectable once
    the dict is detached and the local loss is gone -- the thing the leak kept."""
    import gc

    detach = _load_detach_train_result()
    result, c_loss, saved_activation = _unbackpropped_minibatch()
    del c_loss
    gc.collect()
    assert saved_activation() is not None, "sanity: the dict alone keeps the graph (and its saved activation) alive before detach"

    detach(result)
    gc.collect()
    assert saved_activation() is None, "after detach nothing references the graph, so its saved activation is freed"


def test_all_three_agents_detach_their_train_result():
    """Every agent that builds self.train_result must detach it right after the
    last update(); the leak only bites the distill agent, but the teacher and
    common agents store the same keys."""
    for fname, call in [
        ("common_agent.py", "detach_train_result(self.train_result)"),
        ("intermimic_agent.py", "common_agent.detach_train_result(self.train_result)"),
        ("intermimic_agent_distill.py", "common_agent.detach_train_result(self.train_result)"),
    ]:
        src = open(os.path.join(LEARNING, fname)).read()
        assert call in src, f"{fname} does not detach its train_result"
        # the detach must follow the last update() into the dict, not precede it
        assert src.rfind("self.train_result.update(") < src.rfind(call), fname
