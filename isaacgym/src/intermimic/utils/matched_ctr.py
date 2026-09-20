"""Matched-frame contrastive term for the g3 student (arm `matchctr`), torch only
(no Isaac Gym / rl_games) so the arithmetic is unit-testable on a laptop.

THE IDEA. A body-invariant representation needs positives that are the SAME
MOTION at the SAME MOMENT on DIFFERENT BODIES. Instead of pairing simulators
up front (the twin design, which only re-synced pairs on a coincidental joint
reset and ran at ~5% valid pairs), find them AFTER each rollout in the data
already collected: every sample carries (clip, frame, body); two samples with
equal clip and frame and different body are a positive pair. Negatives are
samples from OTHER clips. Same clip at a different frame is masked out of the
denominator: it is neither the same moment nor a different motion, and pushing
it away would fight the temporal structure of the clip.

COHORT SAMPLING makes those coincidences frequent. Env e owns object
e % n_objects and body e % n_bodies. Group the envs on one object into cohorts
of k consecutive blocks (envs e, e+n_obj, ..., e+(k-1)n_obj -- k different
bodies since consecutive blocks shift the body by n_obj % n_bodies). The lowest
env of a cohort is its LEADER and samples clips as before; the others adopt the
leader's current clip when they reset, with their OWN start frame. Over a
rollout the cohort is therefore on one clip at overlapping frames. On the
activity data every object has exactly one clip, so this changes nothing there
(the sampler already offers one clip per object); on OMOMO (~15 objects, 3356
clips) it lifts exact-frame positives from ~500 to ~5000 per rollout.
"""
import torch


def cohort_leader(env_ids, n_objects, k):
    """Leader env id for each env in `env_ids` (long tensor): the lowest env with
    the same object in the same group of k blocks. k <= 1 -> every env leads."""
    if n_objects <= 0:
        raise ValueError("n_objects must be positive")
    if k <= 1:
        return env_ids.clone()
    block = env_ids // n_objects
    return (block // k) * k * n_objects + env_ids % n_objects


def find_matched_pairs(clip, frame, body, max_anchors, generator=None):
    """(ia, ib): flat sample indices such that clip[ia]==clip[ib], frame[ia]==frame[ib],
    body[ia]!=body[ib]. Up to `max_anchors` anchors are drawn at random; each anchor
    with at least one match gets ONE random partner. clip/frame/body are 1-D long
    tensors of equal length M (a flattened [T, N] rollout). Memory: anchors x M bools."""
    M = clip.shape[0]
    if not (frame.shape[0] == M and body.shape[0] == M):
        raise ValueError("clip, frame, body must have the same length")
    if M == 0 or max_anchors <= 0:
        e = torch.zeros(0, dtype=torch.long, device=clip.device)
        return e, e
    stride = int(frame.max().item()) + 1 if M else 1
    key = clip * stride + frame
    n_a = min(max_anchors, M)
    anchors = torch.randperm(M, generator=generator, device=clip.device)[:n_a]
    same = (key[anchors, None] == key[None, :]) & (body[anchors, None] != body[None, :])
    has = same.any(dim=1)
    if not bool(has.any()):
        e = torch.zeros(0, dtype=torch.long, device=clip.device)
        return e, e
    anchors, same = anchors[has], same[has]
    # one random partner per anchor: random scores, zeroed where not a match, argmax
    score = torch.rand(same.shape, generator=generator, device=clip.device) * same
    partner = score.argmax(dim=1)
    return anchors, partner


def matched_infonce(za, zb, clip_a, tau):
    """Symmetric InfoNCE over P positive pairs. za, zb: [P, D] L2-normalized
    embeddings of the two members of each pair; clip_a: [P] the pair's clip.
    Row i's positive is column i; columns j whose pair is on the SAME clip as i
    (j != i) are masked out of the denominator (same motion, other moment: not a
    negative); every other column is a negative (a different clip). Returns a
    scalar; 0 when P < 2 (nothing to contrast against)."""
    P = za.shape[0]
    if P < 2:
        return torch.zeros((), device=za.device, dtype=za.dtype)
    logits = za @ zb.t() / tau
    same_clip = clip_a[:, None] == clip_a[None, :]
    eye = torch.eye(P, dtype=torch.bool, device=za.device)
    logits = logits.masked_fill(same_clip & ~eye, float("-inf"))
    labels = torch.arange(P, device=za.device)
    ce = torch.nn.functional.cross_entropy
    return 0.5 * (ce(logits, labels) + ce(logits.t(), labels))
