"""InterMimicDistillG4 -- DAgger distillation of the gen4 OBJECT-GROUP teachers.

The g3 task (intermimic_distill_g3.py) picks each env's teacher by its clip's
SOURCE subject. The gen4 object-group teachers (boxes / stands / chairs /
tables, each trained on all 13 OMOMO sources) can only be told apart by the
clip's OBJECT, so this task overrides the three teacher seams of the g3 task
and nothing else:

  * the manifest is OBJECT-keyed (utils/distill_g4.py: `objects: [...]` per
    teacher; a `sources:` manifest is refused -- it belongs to the g3 task);
  * the env -> teacher table is built over the task's sorted object list
    (self.object_name) and indexed by the env's clip object
    (self.object_id[self.data_id], body-major like data_id);
  * startup refuses if any loaded object has no teacher.

Student obs, teacher query, PSI, resets, free-flight gate, the DAgger wrapper:
all inherited from the g3 task unchanged, which inherits the loop from
InterMimic. Fails at startup (not at epoch 1) on a g3 manifest, a missing
teacher file, an object claimed twice, an object with no teacher, plus every
g3 refusal (obs width, token count, horizons).

Env cfg keys: the g3 task's (teacherPolicy, teacherPolicyCFG,
studentObsHorizons, numObsRetarget); teacherPolicy must hold a g4 manifest.
"""
import torch

from .intermimic_distill_g3 import InterMimicDistillG3
from ...utils.distill_g4 import build_object_lookup, load_object_manifest


class InterMimicDistillG4(InterMimicDistillG3):

    # ------------------------------------------------------------------ teacher seams
    def _load_teacher_entries(self, teacher_dir):
        """[ObjectTeacherEntry] from teachers.yaml -- g4: keyed by OBJECT."""
        return load_object_manifest(teacher_dir)

    def _describe_teacher(self, e):
        return f"objects {e.objects}"

    def _build_teacher_routing(self):
        """object -> teacher routing, checked against the data at startup.
        self.object_name is the task's sorted list of loaded object names and
        self.object_id indexes into it per motion (body-major after the
        retarget expansion, like data_id), so the table is aligned with it."""
        names = list(self.object_name)
        lookup, unused = build_object_lookup(self._teacher_entries, names)
        self._object_to_model = torch.tensor(lookup, device=self.device, dtype=torch.long)
        if unused:
            print(f"[distill-g4] NOTE: {len(unused)} teacher(s) serve no clip in this data: {unused}", flush=True)
        print(f"[distill-g4] {len(self._teacher_entries)} teachers cover {len(names)} objects, routed by "
              f"the clip's OBJECT: " + ", ".join(f"{o}->{self._teacher_entries[i].file}"
                                                 for o, i in zip(names, lookup)), flush=True)

    # ------------------------------------------------------------------ routing
    def _refresh_teacher_indices(self):
        """Per-env teacher index from the OBJECT of the env's current clip.
        Called on every reset (data_id changes there) and by the distill agent."""
        obj = self.object_id[self.data_id]
        self.model_indices = self._object_to_model[obj]
        self.sample_indices = torch.arange(self.num_envs, device=self.device)
