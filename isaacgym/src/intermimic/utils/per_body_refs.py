"""Per-body reference files for a one-pair eval (the g3 retarget-tree layout).

The g3 tasks read <tree>/<body>/<clip>.pt: one contact-retargeted reference per
(target body, clip). InterMimic_All (the paper's student) instead reads a FLAT
tree from the authors' release. To score that policy on the same exam as the g3
students -- same clips, same body, same reference -- it needs this layout too.

Kept out of the task module on purpose: env/tasks/* import isaacgym, which a unit
test cannot, and the only thing worth testing here is which files get picked.
"""
import os

from .path_utils import resolve_repo_path


def per_body_reference_files(retarget_dir, motion_file_dir, data_sub, bodies,
                             exists=os.path.exists, listdir=os.listdir,
                             resolve=resolve_repo_path):
    """-> (absolute reference paths, clip basenames) for ONE body, in clip order.

    The clip list comes from motion_file_dir (OMOMO_new: what the g3 students are
    scored on), filtered to the sources in data_sub, NOT from the retarget tree --
    so this policy runs the same clips as the students rather than the authors'
    smaller subset.

    Raises ValueError unless exactly one body is given (an eval runs one
    (body, source) pair; more than one would need the body-major expansion the
    caller does not do), and FileNotFoundError listing what is missing rather
    than letting a caller fall back to some other reference.
    """
    if len(bodies) != 1:
        raise ValueError(
            f"[per_body_refs] retargetedMotionDir needs exactly ONE body in subjectBodies "
            f"(per-pair eval); got {len(bodies)}: {list(bodies)}")
    body = bodies[0]
    clip_names = sorted(n for n in listdir(resolve(motion_file_dir))
                        if n.endswith(".pt") and n.split("_")[0] in data_sub)
    files, missing = [], []
    for cn in clip_names:
        rel = os.path.join(retarget_dir, body, cn)
        p = resolve(rel)
        if exists(p):
            # str(), not the Path resolve_repo_path returns: callers treat these
            # as filenames and split them (intermimic_all.py reads the object
            # name out of '<src>_<object>_<idx>.pt'), which a PosixPath cannot do.
            files.append(str(p))
        else:
            missing.append(rel)
    if missing:
        raise FileNotFoundError(
            f"[per_body_refs] {len(missing)} of {len(clip_names)} (body,clip) reference files "
            f"missing under '{retarget_dir}/{body}', e.g. {missing[:3]}. Generate them "
            f"(scripts/retarget_contact.py --batch); refusing a silent source fallback.")
    return files, clip_names
