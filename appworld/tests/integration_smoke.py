"""Offline API double + real APPWorld: checks orchestration, NOT model quality."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        system = str(payload['messages'][0].get('content', ''))
        if 'Evaluation_Guidelines:' in system:
            content = '{"index": 0, "analysis": "offline protocol test"}'
        elif 'public-evidence selector' in system:
            content = '{"choice":"A","supporting_event_ids":[],"preserve_event_ids":[],"avoid_event_ids":[],"unresolved_issue_type":"NONE","reason":"offline test"}'
        else:
            content = '```python\nprint(apis.supervisor.complete_task(status="fail"))\n```'
        body = json.dumps({'id': 'offline-test', 'choices': [{'message': {'role': 'assistant', 'content': content}}],
                           'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = dict(os.environ, OPENAI_API_KEY='offline-test-not-a-secret',
               OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',
               MODEL_NAME='offline-protocol-test', MAX_RPM='10000',
               OPENAI_REASONING='0', NVIDIA_KEY_POOL='0')
    env.pop('AGENT_EVAL_ENV_FILE', None)
    with tempfile.TemporaryDirectory(prefix='statetrace_smoke_') as tmp:
        command = [sys.executable, str(ROOT / 'run.py'), '--method', 'both', '--split', 'both',
                   '--limit', '1', '--workers', '1', '--max-steps', '1', '--output', tmp]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        for split in ['test_normal', 'test_challenge']:
            directory = Path(tmp) / split
            ids = json.loads((directory / 'experiment.json').read_text())['task_ids']
            for task_id in ids:
                st = directory / 'faithful_v3' / task_id
                oa = directory / 'oagents_candidates' / task_id
                assert (st / 'final.json').exists()
                assert (directory / 'oagents_selector' / f'{task_id}.json').exists()
                assert (st / 'candidate_a.json').read_bytes() == (oa / 'candidate_a.json').read_bytes()
                assert all((oa / f'candidate_{stage}.json').exists() for stage in 'abcd')
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        report = json.loads((Path(tmp) / 'metrics.json').read_text())
        assert all(report[s]['common_tasks'] == 1 for s in report)
        assert all(report[s]['matched']['Vanilla']['SGC'] is None for s in report)
    server.shutdown()
    print('Offline API + real APPWorld two-method/two-split/resume smoke passed.')


if __name__ == '__main__':
    main()
