"""Graph-level regressions; no patient data or anatomical accuracy claims."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bronchial_branch_labeler import BronchialBranchLabeler


def graph_parser(vertices=None, edges=None):
    parser = BronchialBranchLabeler(np.ones((2,2,2),dtype=np.uint8))
    parser.verts = np.asarray(vertices if vertices is not None else [
        (0,0,0), (0,0,20), (-2,0,-3), (2,0,-3),
        (-20,0,-20), (-30,10,-30), (-30,-10,-30),
        (20,0,-20), (30,10,-30), (30,-10,-30)],dtype=float)
    parser.edges = np.asarray(edges if edges is not None else [
        (0,1),(0,2),(0,3),(2,4),(4,5),(4,6),(3,7),(7,8),(7,9)],dtype=int)
    parser._skeletonize = lambda: None
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
        names = [name for _,name in parser.export_items(nodes)]
        self.assertGreaterEqual(len(names),3)
        self.assertTrue(all(name.startswith("Topology_Branch_") for name in names))
        self.assertEqual(set().union(*nodes.values()),set(range(len(parser.verts))))

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


if __name__ == '__main__':
    unittest.main()
