import unittest
import numpy as np
from scripts.audit_bayes_ceiling import (limiting_accuracy_bounds, empirical_conflicts,
    distances_and_neighbors, mst_edges, graph_summary, synthetic_probabilities,
    synthetic_check)


class CeilingAuditTests(unittest.TestCase):
    def test_oracle_moment_bounds(self):
        for eta in (np.array([.5,.5]),np.array([0.,1.]),np.array([.1,.4,.7,.9])):
            r=float(np.mean(2*eta*(1-eta)))
            truth=float(np.maximum(eta,1-eta).mean())
            bounds=limiting_accuracy_bounds(r)
            self.assertLessEqual(bounds['lower'],truth+1e-12)
            self.assertGreaterEqual(bounds['upper'],truth-1e-12)

    def test_out_of_domain_not_silently_clipped(self):
        self.assertFalse(limiting_accuracy_bounds(.51)['admissible'])
        self.assertIsNone(limiting_accuracy_bounds(.51)['upper'])

    def test_duplicate_conflict_is_counted(self):
        x=np.array([[0.],[0.],[1.],[2.]])
        r=empirical_conflicts(x,np.array([0,1,1,0]))
        self.assertEqual(r['empirical_lookup_correct'],3)
        self.assertEqual(r['mixed_groups'],1)
        self.assertEqual(r['singleton_groups'],2)

    def test_graph_preserves_zero_distance_edges_and_excludes_self(self):
        x=np.array([[0.],[0.],[1.],[4.]])
        d,j=distances_and_neighbors(x)
        self.assertTrue(np.all(j!=np.arange(len(x))))
        self.assertEqual(j[0],1)
        self.assertEqual(j[1],0)
        a,b=mst_edges(d)
        self.assertEqual(len(a),3)
        self.assertTrue(any({u,v}=={0,1} for u,v in zip(a,b)))
        r=graph_summary(np.zeros(4,dtype=int),j,(a,b))
        self.assertEqual(r['nn_disagreement'],0)
        self.assertEqual(r['mst_disagreement'],0)

    def test_synthetic_truth_matches_requested_target(self):
        score=np.linspace(-3,3,101)
        for accuracy in (.65,.8,1.):
            p=synthetic_probabilities(score,accuracy)
            self.assertAlmostEqual(float(np.maximum(p,1-p).mean()),accuracy,places=12)

    def test_reversible_scaling_changes_no_exact_collisions(self):
        x=np.array([[0.,1.],[0.,1.],[2.,3.],[4.,0.]])
        y=np.array([0,1,1,0])
        self.assertEqual(empirical_conflicts(x,y),empirical_conflicts(x*np.array([3.,.2])+7,y))

    def test_finite_graph_upper_can_miss_known_deterministic_truth(self):
        x=np.array([[-2.],[-1.],[1.],[2.]])
        d,j=distances_and_neighbors(x)
        edges=mst_edges(d)
        check=synthetic_check(x[:,0],j,edges,repeats=3,targets=(1.,))[0]
        self.assertEqual(check['known_oracle_accuracy'],1.)
        self.assertEqual(check['oracle_moment_bounds']['upper'],1.)
        self.assertLess(check['mst']['upper_mean'],1.)
        self.assertEqual(check['mst']['upper_below_truth_fraction_of_all_repeats'],1.)


if __name__=='__main__':
    unittest.main()
