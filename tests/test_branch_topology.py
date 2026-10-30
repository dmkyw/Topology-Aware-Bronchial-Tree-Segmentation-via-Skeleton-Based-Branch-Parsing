"""Graph-level regressions; no patient data or anatomical accuracy claims."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bronchial_branch_labeler import BronchialBranchLabeler, TopologyStructureError


def graph_parser(vertices=None, edges=None):
    parser = BronchialBranchLabeler(np.ones((2,2,2),dtype=np.uint8))
    parser.verts = np.asarray(vertices if vertices is not None else [
        (0,0,0), (0,0,20), (-2,0,-3), (2,0,-3),
        (-20,0,-20), (-30,10,-30), (-30,-10,-30),
        (20,0,-20), (30,10,-30), (30,-10,-30)],dtype=float)
    parser.edges = np.asarray(edges if edges is not None else [
        (0,1),(0,2),(0,3),(2,4),(4,5),(4,6),(3,7),(7,8),(7,9)],dtype=int)
    parser._skeletonize = lambda const_mm=50.0: None
    parser._build_graph()
    parser._estimate_axes()
    parser._find_carina()
    return parser


class BranchTopologyTests(unittest.TestCase):
    def test_inclined_bronchi_do_not_select_the_terminal_tracheal_limb(self):
        parser = graph_parser()
        a,b = parser._select_main_neighbors()
        self.assertEqual({a,b},{2,3})
        self.assertIsNotNone(parser.first_branch(a,parser.carina))
        self.assertIsNotNone(parser.first_branch(b,parser.carina))

    def test_an_immediate_junction_is_not_skipped(self):
        parser = graph_parser()
        self.assertEqual(parser.first_branch(4,2),4)

    def test_partial_tree_returns_explicit_neutral_topology(self):
        parser = graph_parser()
        nodes = parser.run()
        self.assertEqual(parser.parsing_mode,"TOPOLOGY_ONLY")
        self.assertEqual(parser.diagnostics["selected_skeleton_const_mm"],10.0)
        self.assertEqual(len(parser.diagnostics["anatomy_attempts"]),3)
        names = [name for _,name in parser.export_items(nodes)]
        self.assertGreaterEqual(len(names),3)
        self.assertTrue(all(name.startswith("Topology_Branch_") for name in names))
        self.assertEqual(set().union(*nodes.values()),set(range(len(parser.verts))))

    def test_finer_skeleton_resolves_a_structured_anatomy_limitation(self):
        parser = BronchialBranchLabeler(np.ones((2,2,2),dtype=np.uint8))
        template = graph_parser()
        scales = []
        def skeletonize(const_mm=50.0):
            scales.append(const_mm)
            parser.verts = template.verts.copy()
            parser.edges = template.edges.copy()
            parser.active_scale = const_mm
        parser._skeletonize = skeletonize
        def discover():
            if parser.active_scale == 50.0:
                raise TopologyStructureError("LOBAR_BRANCHES_UNRESOLVED", "coarse scale lacks short paths")
            return {0: set(range(len(parser.verts)))}
        parser._discover_segments = discover
        nodes = parser.run()
        self.assertEqual(scales,[50.0,25.0])
        self.assertEqual(parser.parsing_mode,"ANATOMICAL_CANDIDATES")
        self.assertEqual(parser.diagnostics["selected_skeleton_const_mm"],25.0)
        self.assertEqual(len(nodes[0]),len(parser.verts))

    def test_unexpected_error_does_not_trigger_finer_scale_fallback(self):
        parser = BronchialBranchLabeler(np.ones((2,2,2),dtype=np.uint8))
        template = graph_parser()
        scales = []
        def skeletonize(const_mm=50.0):
            scales.append(const_mm)
            parser.verts = template.verts.copy()
            parser.edges = template.edges.copy()
        parser._skeletonize = skeletonize
        parser._discover_segments = lambda: (_ for _ in ()).throw(RuntimeError("internal error"))
        with self.assertRaisesRegex(RuntimeError,"internal error"):
            parser.run()
        self.assertEqual(scales,[50.0])

    def test_no_branch_points_fails_after_all_physical_scales(self):
        parser = BronchialBranchLabeler(np.ones((2,2,2),dtype=np.uint8))
        scales = []
        def skeletonize(const_mm=50.0):
            scales.append(const_mm)
            parser.verts = np.asarray([(0,0,0),(0,0,1),(0,0,2)],dtype=float)
            parser.edges = np.asarray([(0,1),(1,2)],dtype=int)
        parser._skeletonize = skeletonize
        with self.assertRaises(TopologyStructureError) as caught:
            parser.run()
        self.assertEqual(caught.exception.code,"NO_BRANCH_POINTS")
        self.assertEqual(scales,[50.0,25.0,10.0])

    def test_ras_axes_have_fixed_meaning_and_do_not_depend_on_pca_sign(self):
        parser = graph_parser()
        np.testing.assert_allclose(parser.pc1,(0,0,1))
        np.testing.assert_allclose(parser.pc2,(1,0,0))
        np.testing.assert_allclose(parser.pc3,(0,1,0))

    def test_carina_is_reached_from_the_superior_tracheal_endpoint(self):
        parser = graph_parser()
        self.assertEqual(parser.trachea_root,1)
        self.assertEqual(parser.carina,0)
        self.assertEqual(parser.trace_trachea(),[0,1])

    def test_right_middle_b4_b5_use_lateral_medial_ras_axis(self):
        parser = graph_parser()
        infos = [
            {"nodes": {1}, "x_score": -0.4, "up_score": 0.8, "ap_score": 0.9},
            {"nodes": {2}, "x_score": 0.5, "up_score": -0.8, "ap_score": -0.9},
        ]
        parser.distal_subtrees = lambda *_args, **_kwargs: infos
        named = parser.classify_b45(0, 1, "Rt")
        self.assertEqual(named["Rt_B4"], {2})
        self.assertEqual(named["Rt_B5"], {1})

    def test_left_lingula_b4_b5_use_superior_inferior_ras_axis(self):
        parser = graph_parser()
        infos = [
            {"nodes": {1}, "x_score": 0.8, "up_score": -0.6, "ap_score": 0.9},
            {"nodes": {2}, "x_score": -0.8, "up_score": 0.7, "ap_score": -0.9},
        ]
        parser.distal_subtrees = lambda *_args, **_kwargs: infos
        named = parser.classify_b45(0, 1, "Lt")
        self.assertEqual(named["Lt_B4"], {2})
        self.assertEqual(named["Lt_B5"], {1})

    def test_ambiguous_two_branch_upper_lobe_pattern_fails_to_neutral_fallback(self):
        parser = graph_parser()
        infos = [
            {"nodes": {1}, "seed": 1, "up_score": 0.1, "ap_score": 0.0},
            {"nodes": {2}, "seed": 2, "up_score": 0.05, "ap_score": 0.02},
        ]
        parser.distal_subtrees = lambda *_args, **_kwargs: infos
        with self.assertRaisesRegex(TopologyStructureError, "directional pattern is ambiguous"):
            parser.classify_b123(0, 1, "Rt")


if __name__ == '__main__':
    unittest.main()
