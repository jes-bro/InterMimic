"""Per-object physical properties for an env that MIXES datasets (no torch /
Isaac Gym imports, so tests run locally).

The task's objectMass / objectShapeProps.restitution are ONE value for every
object, which is right when every clip brings the same real object. The
activity student trains on basketball (0.624 kg, bouncy), soccer (0.43 kg) and
a CPR manikin (3.7 kg, inert) in one env, so an env cfg may instead name an
`objectPropsFile`:

    objects:
      bballd03s01rev003b: {mass: 0.624, restitution: 1.0}
      manikin:            {mass: 3.7,   restitution: 0.05}
      ...

written by scripts/merge_activity_data.py from the source arms' cfgs. Every
object the env loads MUST have an entry (refused otherwise -- a default would
silently give a basketball the manikin's bounce); extra entries are allowed.

WHY RESTITUTION IS PER OBJECT WHILE THE PLANE IS NOT. PhysX combines the two
materials of a contact pair by AVERAGE. Each teacher set object and plane
restitution together (bball 0.85/0.85, soccer 0.65/0.65, cpr 0.05/0.7). The
student has ONE plane, so `teacher_pair_to_student_object` solves for the
object value that reproduces each teacher's ball-floor average against the
student's plane: bball 1.0, soccer 0.6, cpr 0.05 at plane 0.7. Ball-body pairs
shift by half the object delta (+0.075 bball, -0.025 soccer); that is the
documented cost of one env.
"""
import yaml


def teacher_pair_to_student_object(obj_r, plane_r, student_plane_r):
    """Object restitution so that (obj + student_plane)/2 == (obj_r + plane_r)/2.
    Refuses a result outside [0, 1] -- pick another student plane value."""
    r = obj_r + plane_r - student_plane_r
    if not (0.0 <= r <= 1.0):
        raise ValueError(
            f"[object-props] teacher pair (obj {obj_r}, plane {plane_r}) needs object restitution "
            f"{r:.3f} against student plane {student_plane_r}, outside [0, 1]")
    return r


def validate_props(props, object_names):
    """`props` = the parsed `objects:` mapping. Returns it; raises on any object
    without an entry, or an entry with a bad mass / restitution."""
    if not isinstance(props, dict) or not props:
        raise ValueError("[object-props] 'objects' must be a non-empty mapping")
    missing = sorted(n for n in object_names if n not in props)
    if missing:
        raise ValueError(f"[object-props] {len(missing)} object(s) have no entry: {missing[:8]}"
                         f"{' ...' if len(missing) > 8 else ''}")
    for name, p in props.items():
        if not isinstance(p, dict) or "mass" not in p or "restitution" not in p:
            raise ValueError(f"[object-props] {name}: entry needs 'mass' and 'restitution', got {p!r}")
        m, r = p["mass"], p["restitution"]
        if not isinstance(m, (int, float)) or isinstance(m, bool) or m <= 0:
            raise ValueError(f"[object-props] {name}: mass must be > 0 kg, got {m!r}")
        if not isinstance(r, (int, float)) or isinstance(r, bool) or not (0.0 <= r <= 1.0):
            raise ValueError(f"[object-props] {name}: restitution must be in [0, 1], got {r!r}")
    return props


def load_object_props(path, object_names):
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    return validate_props(doc.get("objects"), object_names)
