"""Exercise real AppWorld execution/evaluation without paying for any API call."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['APPWORLD_ROOT'] = str(ROOT / 'runtime')


def main():
    from appworld import AppWorld, load_task_ids
    for split in ['test_normal', 'test_challenge']:
        ids = load_task_ids(split)
        with AppWorld(task_id=ids[0], experiment_name='release_environment_smoke', timeout_seconds=None) as world:
            result = world.execute('print(apis.api_docs.show_app_descriptions())')
            assert result and 'Traceback' not in result, result
            evaluation = world.evaluate().to_dict()
            assert isinstance(evaluation['success'], bool)
            print(f'{split}: {len(ids)} tasks; real execution and official evaluator OK')


if __name__ == '__main__':
    main()
