import unittest
import numpy as np
import pandas as pd
from scripts.build_clean_match_dataset import classify, roster_flags, clean, events_for_history, SLOTS
from scripts.team_and_opponent_strength import five


class CleaningRegression(unittest.TestCase):
    def test_winner_counts_not_maximum_length(self):
        # Clean sweep, deciding map, missing map, contradictory map wins, bad BO, unknown BO.
        args=[[3,3,3,3,1,np.nan,5],[2,2,2,2,1,2,3],[0,1,1,0,2,1,0],
              [2,3,2,2,3,3,3],[2,2,2,1,1,2,3]]
        conflict, verified, complete=classify(*(pd.Series(v) for v in args))
        self.assertEqual(complete.tolist(),[True,True,False,False,False,False,True]); self.assertTrue(conflict[4]); self.assertTrue(verified[5])

    def test_conflict_quarantine_and_no_outcome_derived_format(self):
        def row(mid, totals, s1, s2, map_name, game, side2):
            r=dict(match_id=mid,datetime='2025-11-02',team1='a',team2='b',team1_id=1,team2_id=2,tournament='t',bestOf=3,
                   score1_match=s1,score2_match=s2,is_total=totals,map_name=map_name,game_id=game,score1_game=13,score2_game=7)
            for slots,ids in zip(SLOTS,[[1,2,3,4,5],side2]):
                for c,pid in zip(slots,ids): r[c]=pid;r[c[:-3]+'_adr']=70;r[c[:-3]+'_kast']=60
            return r
        rows=[row(1,True,2,0,None,1,[1,7,8,9,10]),row(1,False,2,0,'Dust2',2,[6,7,8,9,10]),
              row(2,True,1,0,None,3,[6,7,8,9,9]),row(2,False,1,0,'Dust2',4,[6,7,8,9,10]),
              row(3,True,2,0,None,5,[6,7,8,9,10]),row(3,False,2,0,'Dust2',6,[6,7,8,9,10]),
              row(3,False,2,0,'Dust2',7,[6,7,8,9,10])];rows[-1]['score2_game']=10
        m,maps,r,rejected=clean(pd.DataFrame(rows),pd.Series({1:'tier1',2:'tier2',3:'tier3'}))
        self.assertEqual(m.bo_reported.tolist(),[3,3,3]); self.assertTrue(m.bo.isna().all()); self.assertEqual(m.bo_score_conflict.tolist(),[False,True,False]); self.assertEqual(len(rejected),2)
        self.assertEqual(set(maps.match_id),{1,2}); self.assertTrue(m.loc[2,'ambiguous_map']); self.assertFalse(m.loc[2,'maps_complete'])
        first,second,_=m.itertuples(index=False); self.assertEqual((five(first,1),five(first,2)),((),())); self.assertEqual(five(second,1),(1,2,3,4,5)); self.assertEqual(five(second,2),())
        e=events_for_history(r,m); self.assertFalse(e[1]); self.assertEqual([p for p,_,_ in e[2]],[1,2,3,4,5]); self.assertEqual(roster_flags([1,None],[2,None]),(False,False,False))


if __name__=='__main__': unittest.main()
