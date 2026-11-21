import unittest
import numpy as np
import pandas as pd

from scripts.roster_and_map_features import past_features, PAIRS, DIFF


class P1HistoryTests(unittest.TestCase):
    def setUp(self):
        self.games=pd.DataFrame({'match_id':[1,2,3,4], 'date':pd.to_datetime(['2025-01-01 08:00','2025-01-01 16:00','2025-01-02 09:00','2025-07-02 12:00']),
                                 'team1':['A']*4,'team2':['B']*4,'team1_won':[1,1,0,0]})
        for side in (1,2):
            for i in range(1,6): self.games[f'team{side}_player{i}_id']=side*10+i
        self.maps=pd.DataFrame({'match_id':[1], 'map_name':['Nuke'],'team1_won_map':[1]})
        self.events={1:[(11,80.,75.)]}

    def test_simultaneous_and_future_inputs_excluded(self):
        ref=past_features(self.games,self.maps,self.events)
        changed=self.games.copy(); changed.loc[0,'team1_won']=0; changed.loc[:1,'team1_player1_id']=99
        alt=self.maps.copy();alt['team1_won_map']=0
        new=past_features(changed,alt,{1:[(99,120.,89.)]})
        pd.testing.assert_frame_equal(ref.iloc[:2],new.iloc[:2])
        for col in ('team1_prior_adr','team1_shared_map_rate','team1_h2h_rate'):
            self.assertNotEqual(ref[col].iloc[2],new[col].iloc[2])
        self.assertEqual(ref.team1_prior_adr.iloc[2],80.)
        self.assertEqual(ref.team1_lineup_run.iloc[2],2)
        self.assertEqual(ref.team1_last_overlap.iloc[2],5)
        self.assertEqual(ref.team1_maps180.iloc[-1],0)
        self.assertEqual(ref.team1_shared_map_rate.iloc[-1],.5)

    def test_swap_changes_only_orientation(self):
        x=past_features(self.games,self.maps,self.events)
        swap=self.games.copy()
        for col in self.games.columns:
            if col.startswith('team1') and col!='team1_won':
                other=col.replace('team1','team2');swap[col]=self.games[other];swap[other]=self.games[col]
        swap.team1_won=1-self.games.team1_won
        maps=self.maps.copy(); maps.team1_won_map=1-maps.team1_won_map
        y=past_features(swap,maps,self.events)
        for left,right in PAIRS: np.testing.assert_allclose(x[left],y[right],equal_nan=True)
        np.testing.assert_allclose(x[DIFF],-y[DIFF],equal_nan=True)
        np.testing.assert_array_equal(x.h2h_games,y.h2h_games)

    def test_incomplete_roster_never_assumed_to_be_a_new_five(self):
        self.games.loc[1,'team1_player1_id']=np.nan
        t=past_features(self.games,self.maps,self.events)
        self.assertEqual(t.team1_lineup_run.iloc[2],1)
        self.assertTrue(np.isnan(t.team1_last_overlap.iloc[2]))
        self.assertEqual(t.team1_players_with_stats.iloc[2],1)
