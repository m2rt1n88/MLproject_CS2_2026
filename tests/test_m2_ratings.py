"""Historical graph features must be causal and independent of team numbering."""
import unittest
import numpy as np
import pandas as pd
from scripts.team_and_opponent_strength import build, EXTRA


class RatingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.matches=pd.DataFrame({'match_id':[1,2,3,4], 'date':pd.to_datetime(['2025-01-01 08:00','2025-01-01 16:00','2025-01-02 10:00','2025-01-09 12:00']),
            'team1':['A','A','B','A'],'team2':['B','C','C','B'],'score1':[2,2,2,0],'score2':[0,0,1,2],
            'tier':['tier1','tier2','tier2','tier1'],'team1_won':[1,1,1,0]})
        for side in (1,2):
            for i in range(1,6):
                self.matches[f'team{side}_player{i}_id']=self.matches[f'team{side}'].map({'A':10,'B':20,'C':30})+i
        self.maps=self.matches[['match_id','score1','score2']].copy()

    def test_scores_and_rosters_arrive_after_utc_day(self):
        original=build(self.matches,self.maps); changed=self.matches.copy(); changed.loc[0,['score1','score2','team1_won','team1_player1_id']]=[0,2,0,99]
        newmaps=self.maps.copy();newmaps.loc[0,['score1','score2']]=[0,13]
        other=build(changed,newmaps);pd.testing.assert_frame_equal(original.iloc[:2],other.iloc[:2]);self.assertNotEqual(original.diff_graph90.iloc[-1],other.diff_graph90.iloc[-1]);
        # Adding later rows cannot alter the historical prefix.
        pd.testing.assert_frame_equal(original.iloc[:3],build(self.matches.iloc[:3],self.maps))

    def test_swap_reverses_all_strength_differences(self):
        original=build(self.matches,self.maps); rotated=self.matches.copy()
        for left,right in [('score1','score2'),('team1','team2')]+[(f'team1_player{i}_id',f'team2_player{i}_id') for i in range(1,6)]:
            rotated[left],rotated[right]=self.matches[right],self.matches[left]
        rotated.team1_won=1-self.matches.team1_won
        alternate=build(rotated,self.maps.rename(columns={'score1':'score2','score2':'score1'}))
        np.testing.assert_allclose(original[EXTRA],-alternate[EXTRA],atol=1e-10)
