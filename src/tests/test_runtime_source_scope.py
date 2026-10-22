"""Source identity follows runtime dependencies, not unrelated optimization work."""
import unittest
from brain_vlm.common import runtime_sources,package_root


class RuntimeSourceScopeTests(unittest.TestCase):
    def test_actual_runtime_dependencies_tracked_and_experiments_excluded(self):
        names={p.relative_to(package_root()).as_posix() for p in runtime_sources()}
        self.assertTrue({'__init__.py','__main__.py','common.py','baseline/pretrained.py','baseline/claims.py',
                         'baseline/vlm/loader.py','refinement/editing.py','viewer.py'} <= names)
        for name in ['optimize/history/image_study.py','optimize/history/crop_data.py','optimize/history/context_study.py','optimize/history/reference_study.py',
                     'optimize/signal_training.py','optimize/link_training.py','optimize/voxel_training.py',
                     'optimize/target_data.py','optimize/region_features.py','optimize/region_cache.py','optimize/region_study.py',
                     'optimize/reference_field.py','optimize/history/roi_study.py']:
            self.assertNotIn(name,names)


if __name__=='__main__':unittest.main()
