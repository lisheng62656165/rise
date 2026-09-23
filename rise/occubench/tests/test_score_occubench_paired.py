from score_occubench_paired import candidates, summarize


def test_final_is_accepted_not_last_proposal():
    result = {'stages': [{'stage': 'A', 'trajectory': 'anchor'},
                         {'stage': 'D', 'trajectory': 'proposal'}],
              'final_stage': 'A', 'final_trajectory': 'anchor'}
    assert candidates(result)['FINAL']['trajectory'] == 'anchor'


def test_paired_denominator_excludes_missing_not_failure():
    labels = {(1, 'A'): {'is_correct': False}, (1, 'FINAL'): {'is_correct': True},
              (2, 'A'): {'is_correct': True}, (3, 'A'): {'is_correct': True},
              (3, 'FINAL'): {'is_correct': False}}
    report = summarize([1, 2, 3], labels)
    assert report['paired_valid'] == 2
    assert report['pending_pairs'] == 1
    assert report['wins'] == report['losses'] == 1
    assert report['delta_pp'] == 0
    assert report['complete'] is False
