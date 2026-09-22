"""Provider-neutral Chat Completions agent used by the release runners.

The historical experiment called this adapter ``MiMoAgent``.  The implementation
is provider-neutral: any OpenAI-compatible endpoint can be selected through the
environment variables consumed by ``reliable_mimo_client.py``.
"""

import sys
from pathlib import Path


# Make direct ``import agents.mimo_agent`` work from the release directory as
# well as from the runners, which normally insert benchmark/ first.
_benchmark = Path(__file__).resolve().parents[1] / "benchmark"
if str(_benchmark) not in sys.path:
    sys.path.insert(0, str(_benchmark))

from .deepseek_flash_agent import DeepSeekFlashAgent


class MiMoAgent(DeepSeekFlashAgent):
    """Backward-compatible name for the bundled Chat Completions adapter."""
