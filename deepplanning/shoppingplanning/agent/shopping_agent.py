"""
Custom Agent implementation - Framework-independent

Uses universal LLM calling for multiple providers
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Lock

_AGENT_EVAL_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / 'src' / 'deepplanning_requirement_ledger.py').is_file()
    ),
    None,
)
if _AGENT_EVAL_ROOT is not None and str(_AGENT_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_EVAL_ROOT))

from src.deepplanning_requirement_ledger import (
    build_requirement_ledger,
    numeric_constraints,
    rank_supported_products,
    remaining_unit_price_window,
    text_for_requirement,
)

try:
    from .call_llm import call_llm
except ImportError:
    from call_llm import call_llm




class ShoppingFnAgent:
    """
    Lightweight function-calling Agent (shopping scenario):
    - Loads shopping_tool_schema.json as OpenAI Chat Completions tools
    - Dynamically loads tool classes (BaseShoppingTool subclasses) from shopping_tools directory
    - Iteratively calls LLM and executes tool_calls until final answer
    """

    def __init__(self,
                 model: str | None = None,
                 tool_schema_path: str | None = None,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 sample_id: str | None = None,
                 database_base_path: str | None = None) -> None:
        """
        Initialize Agent
        
        Args:
            model: Model name (must exist in models_config.json)
            tool_schema_path: Path to tool schema JSON file
            base_url: Base URL for API (deprecated, loaded from models_config.json)
            api_key: API key (deprecated, loaded from models_config.json)
            sample_id: Sample ID for database path resolution
            database_base_path: Base path to database directory
        """
        self._load_env_from_dotenv()

        self.model = model or os.getenv("TOOLS_AGENT_MODEL", "qwen-plus")
        default_schema = Path(__file__).resolve().parent / 'tools' / 'shopping_tool_schema.json'
        self.tool_schema_path = tool_schema_path or os.getenv("SHOPPING_SCHEMA_PATH", str(default_schema))

        self.sample_id = sample_id
        if database_base_path:
            self.database_base_path = Path(database_base_path)
        else:
            # Default path: ShoppingBench/database
            project_root = Path(__file__).resolve().parent
            self.database_base_path = project_root / 'database'

        self.tool_config = self._build_tool_config()
        self.tools_schema = self._load_tool_schemas()
        self.openai_tools = self._build_openai_tools(self.tools_schema)
        self.tool_instances = self._load_tool_instances()

        if not Path(self.tool_schema_path).exists():
            raise FileNotFoundError(f"Tool schema not found: {self.tool_schema_path}")

    def _build_tool_config(self) -> Dict[str, Any]:
        """
        Build tool configuration with database path.
        All shopping tools use the same products.jsonl file, simplifying the logic.
        """
        cfg = {}
        if self.sample_id is not None:
            # Shopping scenario database path structure: database/case_{sample_id}/products.jsonl
            db_path = self.database_base_path / f'case_{self.sample_id}'
            
            if db_path.exists():
                cfg['database_path'] = str(db_path)
            else:
                if os.getenv('DEBUG_TOOLS') == '1':
                    print(f"[ShoppingFnAgent] WARN: Database not found for case {self.sample_id}: {db_path}")
        return cfg
    
    def _load_tool_instances(self) -> Dict[str, Any]:
        """
        Dynamically load tool instances from TOOL_REGISTRY.
        
        Tool registration mechanism:
        1. Tool classes use the @register_tool('tool_name') decorator
        2. The decorator executes at class definition time, registering the tool class to base_shopping_tool.TOOL_REGISTRY
        3. When importing the tools package, __init__.py imports all tool modules, triggering decorator execution
        4. Retrieve registered tool classes from TOOL_REGISTRY and instantiate them
        """
        instances: Dict[str, Any] = {}

        tools_dir = Path(__file__).resolve().parent.parent / 'tools'
        # Add tools_dir to sys.path to enable 'from base_shopping_tool import ...' in tool files
        sys.path.insert(0, str(tools_dir))
        sys.path.insert(0, str(tools_dir.parent))

        # Import tools package to trigger @register_tool decorator execution for all tool modules
        # tools/__init__.py imports all tool modules, and decorators register tool classes to TOOL_REGISTRY
        try:
            import tools  # noqa: F401
        except Exception as e:
            if os.getenv('DEBUG_TOOLS') == '1':
                print(f"[ShoppingFnAgent] WARN: import tools failed: {e}")
            return instances

        # Get TOOL_REGISTRY from base_shopping_tool module
        try:
            import base_shopping_tool  # type: ignore
            tool_registry = getattr(base_shopping_tool, 'TOOL_REGISTRY', None)
            if tool_registry is None:
                if os.getenv('DEBUG_TOOLS') == '1':
                    print("[ShoppingFnAgent] WARN: TOOL_REGISTRY not found in base_shopping_tool")
                return instances
        except Exception as e:
            if os.getenv('DEBUG_TOOLS') == '1':
                print(f"[ShoppingFnAgent] WARN: import base_shopping_tool failed: {e}")
            return instances

        if not tool_registry:
            print("[ShoppingFnAgent] WARN: TOOL_REGISTRY is empty. No tools were registered.")
            return instances

        # Create tool instances from TOOL_REGISTRY
        tool_cfg = self.tool_config
        for tool_name, tool_cls in tool_registry.items():
            try:
                inst = tool_cls(cfg=tool_cfg)
                instances[tool_name] = inst
            except Exception as e:
                if os.getenv('DEBUG_TOOLS') == '1':
                    print(f"[ShoppingFnAgent] WARN: Failed to instantiate tool '{tool_name}': {e}")
                continue

        return instances

    def _load_env_from_dotenv(self) -> None:
        """
        Load environment variables from .env file
        
        Searches for .env in the following order:
        1. Domain directory (shoppingplanning/)
        2. Project root (parent of domain)
        """
        try:
            # Try domain directory first
            domain_root = Path(__file__).resolve().parent.parent
            domain_dotenv = domain_root / '.env'
            
            # Try project root
            project_root = domain_root.parent
            project_dotenv = project_root / '.env'
            
            # Use project root .env if it exists, otherwise domain .env
            dotenv_path = project_dotenv if project_dotenv.exists() else domain_dotenv
            
            if not dotenv_path.exists():
                return
            
            for line in dotenv_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and (key not in os.environ):
                    os.environ[key] = val
        except Exception:
            pass

    def _load_tool_schemas(self) -> List[Dict[str, Any]]:
        """Load tool schemas from JSON file"""
        with open(self.tool_schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_openai_tools(self, schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build OpenAI tools format
        - If schema is already {type:function, function:{...}}, use as-is
        - Otherwise wrap as function definition
        """
        tools: List[Dict[str, Any]] = []
        for s in schemas:
            if isinstance(s, dict) and s.get('type') == 'function' and isinstance(s.get('function'), dict):
                tools.append(s)
        return tools

    def _exec_tool(self, name: str, arguments_json: str) -> str:
        """Execute tool call"""
        inst = self.tool_instances.get(name)
        if not inst:
            return json.dumps({"error": f"tool '{name}' not found"}, ensure_ascii=False)
        try:
            res = inst.call(arguments_json)  # Pass raw JSON string
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        config_name: Optional[str] = None,
        max_tokens_override: Optional[int] = None,
    ):
        """Call LLM with unified handling for all models"""
        return call_llm(
            config_name=config_name or self.model,
            messages=messages,
            tools=tools,
            max_tokens_override=max_tokens_override,
        )

    def _detect_tool_calls(self, assistant_message) -> List[Dict[str, Any]]:
        """Detect and normalize tool calls"""
        tool_calls = getattr(assistant_message, 'tool_calls', None)
        calls: List[Dict[str, Any]] = []
        self._last_text_tool_call_fallback = False
        self._last_text_tool_call_raw = None
        if not tool_calls:
            if not self._persist_proposed_v3_enabled():
                return calls
            content = str(getattr(assistant_message, 'content', None) or '')
            valid_tools = set(getattr(self, 'tool_instances', {}))
            for block in re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, flags=re.DOTALL):
                try:
                    payload = json.loads(block)
                except Exception:
                    continue
                name = payload.get('name')
                arguments = payload.get('arguments', {})
                if name not in valid_tools or not isinstance(arguments, dict):
                    continue
                calls.append({
                    'id': f"call_{uuid.uuid4().hex[:24]}",
                    'name': name,
                    'arguments': json.dumps(arguments, ensure_ascii=False),
                })
            self._last_text_tool_call_fallback = bool(calls)
            if calls:
                self._last_text_tool_call_raw = content
            return calls
        
        for idx, tc in enumerate(tool_calls):
            try:
                # Generate unique ID if not provided by the model
                tool_call_id = tc.id
                if tool_call_id is None or not tool_call_id:
                    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                
                calls.append({
                    'id': tool_call_id,
                    'name': tc.function.name,
                    'arguments': tc.function.arguments,
                })
            except Exception:
                continue
        
        return calls

    def _add_to_cart(self, history_messages: List[Any]) -> List[Any]:
        history_messages = list(history_messages)
        history_messages.append({
            "role": "user",
            "content": (
                "Check whether the items in the shopping cart meet the requirements. "
                "If not, add the required items to the cart. If there are multiple possible solutions, "
                "choose the optimal one. The final result should be based on the items in the cart. "
                "If the task is already complete, then stop."
            )
        })
        return history_messages

    def run(self, user_query: str, system_prompt: str | None = None, max_llm_calls: int = 100, save_messages: bool = True, messages_output_dir: str | None = None, sample_id: str | None = None) -> List[Any]:
        """
        Agent main loop: Call LLM → Execute tools → Repeat until final answer
        
        Args:
            user_query: User query
            system_prompt: System prompt
            max_llm_calls: Maximum LLM calls
            save_messages: Whether to save messages to file
            messages_output_dir: Output directory for messages (if sample_id not provided)
            sample_id: Sample ID for database path resolution
            
        Returns:
            Complete message history
        """
        if save_messages:
            # If sample_id exists, save to {database_base_path}/case_{sample_id}/messages.json
            # Use self.database_base_path for proper isolation when running concurrent instances
            if sample_id:
                db_case_dir = self.database_base_path / f'case_{sample_id}'
                db_case_dir.mkdir(parents=True, exist_ok=True)
                messages_file = db_case_dir / 'messages.json'
            else:
                # Otherwise fallback to result/messages
                msg_dir = Path(messages_output_dir or (Path(__file__).resolve().parent.parent / 'result' / 'messages'))
                msg_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                messages_file = msg_dir / f'messages_{ts}.json'

        messages: List[Any] = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": user_query}]
        persist_audit_file = messages_file.with_name('persist_audit.json') if save_messages else None
        persist_state = self._persist_new_state(persist_audit_file)
        persist_state['sample_id'] = str(sample_id or self.sample_id or '')
        persist_state['user_query'] = user_query
        self._persist_load_v5_ledger(persist_state, messages_file)
        self._persist_load_pfl_card(persist_state, messages_file)
        if save_messages:
            self._save_messages(messages, messages_file, 0, "Initial messages")
            self._persist_save_audit(persist_state)

        for step_count in range(1, max_llm_calls + 1):
            if self._persist_context_budget_exceeded(messages) and not self._persist_proposed_v3_enabled() and not self._persist_pfl_plan_enabled():
                messages.append({"role": "assistant", "content": self._persist_context_budget_message('search')})
                persist_state['events'].append({'type': 'intervention', 'phase': 'search', 'step': step_count, 'reason': 'shopping_context_budget_finalization'})
                if save_messages:
                    self._save_messages(messages, messages_file, step_count, "PERSIST context budget finalization")
                    self._persist_save_audit(persist_state)
                break
            if self._persist_pfl_r4_search_budget_exceeded(persist_state, messages, step_count):
                messages.append({
                    'role': 'system',
                    'content': (
                        '[PERSIST-ACE-Plan r4] Retrieval has consumed the online overflow budget. '
                        'Do not start more broad search/detail loops. Use visible evidence and proceed to cart verification/repair.'
                    ),
                })
                persist_state['events'].append({
                    'type': 'pfl_r4_budget_transition',
                    'phase': 'search',
                    'step': step_count,
                    'reason': 'overflow_prone_search_budget_to_cart_check',
                    'history_chars': self._persist_estimate_history_chars(messages),
                })
                if save_messages:
                    self._save_messages(messages, messages_file, step_count, "PFL r4 search budget transition")
                    self._persist_save_audit(persist_state)
                break
            llm_messages = self._persist_messages_for_llm(persist_state, messages, 'search', step_count)
            resp = self._persist_v5_call_with_overflow_recovery(
                persist_state, llm_messages, self.openai_tools, 'search', step_count
            )
            msg = resp.choices[0].message
            
            # Convert message object to serializable dict
            msg_dict = {
                "role": "assistant",
                "content": msg.content or '',
            }
            
            # Preserve reasoning_content if present
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                msg_dict['reasoning_content'] = msg.reasoning_content
            
            calls = self._detect_tool_calls(msg)
            if getattr(self, '_last_text_tool_call_fallback', False):
                persist_state['events'].append({'type': 'intervention', 'phase': 'search', 'step': step_count, 'reason': 'shopping_text_tool_call_fallback', 'recovered_calls': len(calls), 'raw_response': self._last_text_tool_call_raw})
                msg_dict['content'] = re.sub(r'<tool_call>\s*\{.*?\}\s*</tool_call>', '', msg_dict['content'], flags=re.DOTALL).strip()
            calls, persist_message = self._persist_prepare_tool_calls(persist_state, calls, 'search', step_count)
            calls = self._persist_apply_current_turn_fanout_gate(persist_state, calls, 'search', step_count)
            if persist_message:
                msg_dict['content'] = persist_message
            if calls:
                msg_dict["tool_calls"] = [
                    {
                        'id': call['id'],
                        'type': 'function',
                        'function': {
                            'name': call['name'],
                            'arguments': call['arguments']
                        }
                    }
                    for call in calls
                ]
            
            messages.append(msg_dict)
            if save_messages:
                self._save_messages(messages, messages_file, step_count, f"LLM response - {len(calls)} tool calls")
                self._persist_save_audit(persist_state)
            
            if not calls:
                break

            for call in calls:
                tool_result = self._exec_tool(call['name'], call['arguments'])
                self._persist_record_result(persist_state, call, tool_result, 'search', step_count)
                messages.append({"role": "tool", "tool_call_id": call['id'], "content": tool_result})
            if save_messages:
                self._save_messages(messages, messages_file, step_count, f"Tool execution completed - {len(calls)} tools")
                self._persist_save_audit(persist_state)

        messages = self._add_to_cart(messages)
        for step_count in range(1, max_llm_calls + 1):
            if self._persist_context_budget_exceeded(messages) and not self._persist_proposed_v3_enabled() and not self._persist_pfl_plan_enabled():
                messages.append({"role": "assistant", "content": self._persist_context_budget_message('cart_check')})
                persist_state['events'].append({'type': 'intervention', 'phase': 'cart_check', 'step': step_count, 'reason': 'shopping_context_budget_finalization'})
                if save_messages:
                    self._save_messages(messages, messages_file, step_count, "PERSIST context budget finalization")
                    self._persist_save_audit(persist_state)
                return messages
            if self._persist_pfl_r4_cart_budget_exceeded(persist_state, messages, step_count):
                messages.append({
                    'role': 'assistant',
                    'content': (
                        '[PERSIST-ACE-Plan r4] Cart verification reached the overflow budget. '
                        'I will stop further tool calls and leave the current evidence-backed cart as the final state.'
                    ),
                })
                persist_state['events'].append({
                    'type': 'pfl_r4_budget_transition',
                    'phase': 'cart_check',
                    'step': step_count,
                    'reason': 'overflow_prone_cart_check_budget_finish',
                    'history_chars': self._persist_estimate_history_chars(messages),
                })
                if save_messages:
                    self._save_messages(messages, messages_file, step_count, "PFL r4 cart budget finish")
                    self._persist_save_audit(persist_state)
                return messages
            llm_messages = self._persist_messages_for_llm(persist_state, messages, 'cart_check', step_count)
            resp = self._persist_v5_call_with_overflow_recovery(
                persist_state, llm_messages, self.openai_tools, 'cart_check', step_count
            )
            msg = resp.choices[0].message
            
            # Convert message object to serializable dict
            msg_dict = {
                "role": "assistant",
                "content": msg.content or '',
            }
            
            # Preserve reasoning_content if present
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                msg_dict['reasoning_content'] = msg.reasoning_content
            
            calls = self._detect_tool_calls(msg)
            if getattr(self, '_last_text_tool_call_fallback', False):
                persist_state['events'].append({'type': 'intervention', 'phase': 'cart_check', 'step': step_count, 'reason': 'shopping_text_tool_call_fallback', 'recovered_calls': len(calls), 'raw_response': self._last_text_tool_call_raw})
                msg_dict['content'] = re.sub(r'<tool_call>\s*\{.*?\}\s*</tool_call>', '', msg_dict['content'], flags=re.DOTALL).strip()
            calls, persist_message = self._persist_prepare_tool_calls(persist_state, calls, 'cart_check', step_count)
            calls = self._persist_apply_current_turn_fanout_gate(persist_state, calls, 'cart_check', step_count)
            if persist_message:
                msg_dict['content'] = persist_message
            if calls:
                msg_dict["tool_calls"] = [
                    {
                        'id': call['id'],
                        'type': 'function',
                        'function': {
                            'name': call['name'],
                            'arguments': call['arguments']
                        }
                    }
                    for call in calls
                ]
            
            messages.append(msg_dict)
            if save_messages:
                self._save_messages(messages, messages_file, step_count, f"LLM response - {len(calls)} tool calls")
                self._persist_save_audit(persist_state)
            
            if not calls:
                if (
                    persist_message
                    and persist_state.get('pfl_card_prompted')
                    and persist_state.get('v5_active_closure_subtype')
                    and int(persist_state.get('pfl_state_guard_reprompts', 0)) < 1
                ):
                    persist_state['pfl_state_guard_reprompts'] = 1
                    serialized_reset_blocked = any(
                        event.get('reason') == 'require_global_range_reset_before_serialized_closure'
                        for event in persist_state.get('events', [])[-3:]
                    )
                    new_planning_recovery = persist_state.get('v5_trigger_closure_subtype') in {
                        'REPEATED_CART_CHECK_MISSING_EVIDENCE',
                        'COUPON_OSCILLATION_MISSING_PREREQUISITE',
                    }
                    messages.append({
                        'role': 'system',
                        'content': (
                            '[PERSIST-ACE-Plan serialized closure] Do not finish yet. Issue exactly one '
                            'filter_by_range call without product_ids, using one condition_key from '
                            'recovery_contract.allowed_range_keys. Wait for that real result before any next filter.'
                            if serialized_reset_blocked
                            else
                            '[PERSIST-ACE-Planning requirement guard] Do not repeat the previous numeric range '
                            'lookup. Re-read the sole unresolved requirement in Card JSON and switch to one '
                            'still-unused constraint family: use search_products for the product identity, or '
                            'filter_by_brand, filter_by_color, or filter_by_size on the real IDs returned by the '
                            'previous lookup. Use no invented ID and wait for the real result.'
                            if new_planning_recovery
                            else
                            '[PERSIST-ACE-Plan requirement guard] The previous proposed lookup was blocked '
                            'because it repeated without progress or used a condition outside the sole unresolved '
                            'requirement. Use one different valid condition_key from the recovery_contract in Card JSON, '
                            'or finish if no evidence-backed action remains.'
                        ),
                    })
                    persist_state['events'].append({
                        'type': 'pfl_state_guard_reprompt',
                        'phase': 'cart_check',
                        'step': step_count,
                        'reason': 'one_time_reprompt_after_requirement_scoped_guard',
                    })
                    if save_messages:
                        self._save_messages(messages, messages_file, step_count, 'PERSIST state guard reprompt')
                        self._persist_save_audit(persist_state)
                    continue
                premature_finalize_message = self._persist_premature_finalize_message(
                    persist_state,
                    'cart_check',
                    step_count,
                )
                if premature_finalize_message:
                    messages.append({
                        'role': 'system',
                        'content': premature_finalize_message,
                    })
                    if save_messages:
                        self._save_messages(
                            messages,
                            messages_file,
                            step_count,
                            'PERSIST premature finalize recovery',
                        )
                        self._persist_save_audit(persist_state)
                    continue
                if self._persist_should_reprompt_cart_completion(persist_state, msg_dict.get('content', '')):
                    persist_state['cart_completion_reprompts'] = int(persist_state.get('cart_completion_reprompts', 0)) + 1
                    messages.append({
                        'role': 'user',
                        'content': (
                            '[PERSIST-ACE] shopping_cart_completion_card: your visible cart review explicitly identified '
                            'one or more missing or non-compliant requirements. Repair only those identified gaps using '
                            'visible product IDs and tools. Preserve supported cart items, do not invent IDs, then verify '
                            'the cart once and finish.'
                        ),
                    })
                    persist_state['events'].append({
                        'type': 'intervention',
                        'phase': 'cart_check',
                        'step': step_count,
                        'reason': 'shopping_cart_completion_reprompt',
                        'reprompt_count': persist_state['cart_completion_reprompts'],
                    })
                    if save_messages:
                        self._save_messages(messages, messages_file, step_count, 'PERSIST cart completion reprompt')
                        self._persist_save_audit(persist_state)
                    continue
                return messages

            for call in calls:
                tool_result = self._exec_tool(call['name'], call['arguments'])
                self._persist_record_result(persist_state, call, tool_result, 'cart_check', step_count)
                messages.append({"role": "tool", "tool_call_id": call['id'], "content": tool_result})
            if save_messages:
                self._save_messages(messages, messages_file, step_count, f"Tool execution completed - {len(calls)} tools")
                self._persist_save_audit(persist_state)

        return messages
    

    _PERSIST_LOOKUP_TOOLS = {
        'search_products',
        'filter_by_brand',
        'filter_by_color',
        'filter_by_size',
        'filter_by_applicable_coupons',
        'filter_by_range',
        'sort_products',
        'get_product_details',
        'calculate_transport_time',
        'get_user_info',
        'get_cart_info',
    }
    _PERSIST_ACTION_TOOLS = {
        'add_product_to_cart',
        'delete_product_from_cart',
        'add_coupon_to_cart',
        'delete_coupon_from_cart',
    }
    _PERSIST_RATING_KEY_ALIASES = {
        f'rating.{prefix}{suffix}': f'rating.distribution.{number}_star'
        for number, prefix in (
            (1, 'one_'),
            (2, 'two_'),
            (3, 'three_'),
            (4, 'four_'),
            (5, 'five_'),
        )
        for suffix in ('star', 'star_reviews')
    } | {
        f'rating.total_{prefix}{suffix}': f'rating.distribution.{number}_star'
        for number, prefix in (
            (1, 'one_'),
            (2, 'two_'),
            (3, 'three_'),
            (4, 'four_'),
            (5, 'five_'),
        )
        for suffix in ('star', 'star_reviews')
    }
    _PERSIST_COUPON_PATTERN = re.compile(
        r'(?:Cross-store|Same-brand|VIP):\s*\D*\d[\d,]*\s+off\s+every\s+\D*\d[\d,]*',
        re.IGNORECASE,
    )

    def _persist_guard_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return mode not in {'', '0', 'false', 'none', 'off'}

    def _persist_load_action_registry(self) -> Dict[str, Any]:
        registry_path = os.getenv('SHOPPING_PERSIST_ACTION_REGISTRY_PATH', '').strip()
        if not registry_path:
            return {}
        with open(registry_path, 'r', encoding='utf-8') as registry_file:
            registry = json.load(registry_file)
        cards = [
            card for card in registry.get('cards', [])
            if card.get('enabled', True) and card.get('card_id')
        ]
        cards.sort(key=lambda card: (int(card.get('priority', 999)), str(card['card_id'])))
        return {
            'registry_id': str(registry.get('registry_id') or Path(registry_path).stem),
            'registry_path': registry_path,
            'cards': cards,
        }

    def _persist_ablation_mode(self) -> str:
        return os.getenv('SHOPPING_PERSIST_ABLATION', '').strip().lower()

    def _persist_new_state(self, audit_file: Path | None = None) -> Dict[str, Any]:
        action_registry = self._persist_load_action_registry()
        return {
            'enabled': self._persist_guard_enabled() or bool(action_registry.get('cards')),
            'audit_file': str(audit_file) if audit_file else None,
            'events': [],
            'action_registry': action_registry,
            'action_registry_triggered_cards': set(),
            'coupon_aliases_from_prior_errors': {},
            'failed_lookup_counts': {},
            'authoritative_failed_lookup_keys': set(),
            'failed_action_keys': set(),
            'failed_action_versions': {},
            'failed_action_tool_counts': {},
            'evidence_version': 0,
            'failed_evidence_versions': {},
            'no_effect_lookup_counts': {},
            'no_effect_evidence_versions': {},
            'pfl_lookup_stale_run': 0,
            'pfl_global_llm_step': 0,
            'visible_product_ids': set(),
            'detail_product_ids': set(),
            'latest_cart_product_ids': set(),
            'latest_cart_total': None,
            'nonempty_cart_seen': False,
            'delete_calls_after_nonempty_cart': 0,
            'pfl_coupon_completion_done': False,
            'serialized_closure_lookup_count': 0,
            'lookup_signature_counts': {},
            'last_lookup_failed': None,
            'last_retrieval_product_ids': set(),
            'observed_evidence_signatures': set(),
            'context_compaction_signatures': set(),
            'context_compaction_counts': {},
            'cart_completion_reprompts': 0,
            'v5_shadow_pending': {},
            'v5_requirement_ledger': None,
            'v5_requirement_products': {},
            'v5_requirement_transport_times': {},
            'v5_requirement_state': None,
            'v5_requirement_last_delta': {},
            'v5_requirement_satisfaction_stale_run': 0,
            'v5_requirement_recent_satisfaction_history': [],
            'v5_requirement_recent_satisfaction_churn': 0,
            'v5_pending_ace': None,
            'v5_ace_prompt_count': 0,
            'v5_ledger_prompt_count': 0,
            'v5_compaction_counts': {},
            'pfl_card': None,
            'pfl_card_prompted': False,
            'pfl_card_shadow_only': False,
            'pfl_card_shadow_logged': False,
            'pfl_handoff_gate_logged': False,
            'pfl_handoff_gate_signature': None,
            'last_executed_call_key': None,
            'last_executed_call_tool': None,
        }

    def _persist_normalize_arguments(self, arguments_json: str) -> str:
        try:
            parsed = json.loads(arguments_json or '{}')
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        except Exception:
            return str(arguments_json or '')

    def _persist_call_key(self, call: Dict[str, Any]) -> str:
        return f"{call.get('name')}::{self._persist_normalize_arguments(call.get('arguments', ''))}"

    def _persist_normalize_text(self, value: Any) -> str:
        return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())

    def _persist_product_name_like(self, value: Any) -> bool:
        text = str(value or '').strip()
        normalized = self._persist_normalize_text(text)
        if (
            not text
            or normalized in {'id', 'product', 'productid', 'product123', '12345'}
            or re.fullmatch(r'[a-f0-9]{8}', text.lower())
            or re.fullmatch(r'<?product[_-]?id(?:[_-]?\d+)?>?', text, re.IGNORECASE)
            or re.fullmatch(r'<?product(?:[_-]?\d+)?>?', text, re.IGNORECASE)
        ):
            return False
        if len(text) < 5 or len(text) > 160 or not re.search(r'[A-Za-z]', text):
            return False
        return bool(re.search(r'[\s_-]', text) or len(re.findall(r'[A-Za-z]+', text)) >= 2)

    def _persist_coupon_aliases_from_error(self, result: str) -> Dict[str, str]:
        text = str(result or '')
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                text = str(parsed.get('error') or text)
        except Exception:
            pass
        if 'valid coupons are' not in text.lower():
            return {}
        return {
            self._persist_normalize_text(match.group(0)): match.group(0)
            for match in self._PERSIST_COUPON_PATTERN.finditer(text)
        }

    def _persist_result_failed(self, result: str) -> bool:
        text = str(result or '').strip()
        lower = text.lower()
        if not text:
            return True
        if any(marker in lower for marker in ['"error"', 'traceback', 'exception', 'failed', 'not found', 'invalid', 'cannot', 'unable']):
            return True
        try:
            parsed = json.loads(text)
        except Exception:
            return False
        if isinstance(parsed, dict):
            if parsed.get('error') or parsed.get('success') is False or parsed.get('status') in {'error', 'failed', 'fail'}:
                return True
            if 'results' in parsed and parsed.get('results') in ([], None):
                return True
            if 'products' in parsed and parsed.get('products') in ([], None):
                return True
        if isinstance(parsed, list) and len(parsed) == 0:
            return True
        return False

    def _persist_authoritative_lookup_error(self, result: str) -> bool:
        lower = str(result or '').strip().lower()
        return 'condition_key cannot be found in product data' in lower

    def _persist_refresh_requirement_state(self, state: Dict[str, Any], result: str = '') -> None:
        requirements = state.get('v5_requirement_ledger') or []
        if not requirements:
            return
        try:
            parsed = json.loads(result) if result else None
        except Exception:
            parsed = None
        products = state.setdefault('v5_requirement_products', {})
        transport_times = state.setdefault('v5_requirement_transport_times', {})
        if isinstance(parsed, dict):
            if isinstance(parsed.get('products'), list):
                for product in parsed['products']:
                    if isinstance(product, dict) and product.get('product_id'):
                        products[str(product['product_id'])] = product
            if parsed.get('product_id') and parsed.get('estimated_delivery_days') is not None:
                try:
                    transport_times[str(parsed['product_id'])] = int(parsed['estimated_delivery_days'])
                except (TypeError, ValueError):
                    pass
        previous = state.get('v5_requirement_state') or {}
        current = build_requirement_ledger(
            requirements=requirements,
            visible_ids=state.get('visible_product_ids') or set(),
            products=products,
            cart_ids=state.get('latest_cart_product_ids') or set(),
            transport_times=transport_times,
        )
        delta = {
            field: int(current.get(field, 0)) - int(previous.get(field, 0))
            for field in ('solved_count', 'verified_count', 'supported_candidate_count')
        }
        delta['missing_requirement_count'] = (
            int(current.get('missing_requirement_count', 0))
            - int(previous.get('missing_requirement_count', current.get('missing_requirement_count', 0)))
        )
        satisfaction_progress = bool(previous) and (
            delta['solved_count'] > 0
            or delta['verified_count'] > 0
            or delta['supported_candidate_count'] > 0
            or delta['missing_requirement_count'] < 0
        )
        if not previous or satisfaction_progress:
            state['v5_requirement_satisfaction_stale_run'] = 0
        else:
            state['v5_requirement_satisfaction_stale_run'] = int(
                state.get('v5_requirement_satisfaction_stale_run', 0)
            ) + 1
        history = list(state.get('v5_requirement_recent_satisfaction_history') or [])
        history.append(1 if satisfaction_progress else 0)
        history = history[-6:]
        state['v5_requirement_recent_satisfaction_history'] = history
        state['v5_requirement_recent_satisfaction_churn'] = sum(history)
        state['v5_requirement_last_delta'] = delta
        state['v5_requirement_state'] = current

    def _persist_record_result(self, state: Dict[str, Any], call: Dict[str, Any], result: str, phase: str, step: int) -> None:
        if not state.get('enabled'):
            return
        key = self._persist_call_key(call)
        name = call.get('name')
        state['last_executed_call_key'] = key
        state['last_executed_call_tool'] = name
        failed = self._persist_result_failed(result)
        state['coupon_aliases_from_prior_errors'].update(
            self._persist_coupon_aliases_from_error(result)
        )
        product_ids, cart_product_ids = self._persist_extract_result_evidence(call.get('name'), result)
        try:
            parsed_result = json.loads(result)
        except Exception:
            parsed_result = None
        if (
            isinstance(parsed_result, dict)
            and isinstance(parsed_result.get('items'), list)
            and isinstance(parsed_result.get('summary'), dict)
        ):
            state['latest_cart_total'] = parsed_result['summary'].get('total_price')
        previous_products = set(state.get('visible_product_ids', set()))
        previous_cart = set(state.get('latest_cart_product_ids', set()))
        result_signature = hashlib.sha256(f'{name}\n{str(result or "").strip()}'.encode('utf-8')).hexdigest()
        result_evidence_changed = not failed and result_signature not in state['observed_evidence_signatures']
        if not failed:
            state['observed_evidence_signatures'].add(result_signature)
        state['visible_product_ids'].update(product_ids)
        if (
            not failed
            and name in {
                'search_products',
                'filter_products',
                'filter_by_brand',
                'filter_by_color',
                'filter_by_size',
                'filter_by_range',
                'sort_products',
            }
        ):
            state['last_retrieval_product_ids'] = set(product_ids)
        if name == 'get_product_details':
            state['detail_product_ids'].update(product_ids)
        if cart_product_ids is not None:
            state['latest_cart_product_ids'] = set(cart_product_ids)
        if previous_cart or state['latest_cart_product_ids']:
            state['nonempty_cart_seen'] = True
        if name == 'delete_product_from_cart' and state.get('nonempty_cart_seen'):
            state['delete_calls_after_nonempty_cart'] = int(
                state.get('delete_calls_after_nonempty_cart', 0)
            ) + 1
        self._persist_refresh_requirement_state(state, result)
        if (
            name == 'add_coupon_to_cart'
            and not failed
            and state.get('pfl_card_prompted')
            and state.get('v5_active_closure_subtype')
            and int((state.get('v5_requirement_state') or {}).get('missing_requirement_count', 1)) == 0
        ):
            state['pfl_coupon_completion_done'] = True
            state['events'].append({
                'type': 'pfl_coupon_completion_checkpoint',
                'phase': phase,
                'step': step,
                'decision': 'freeze_coupon_mutations_after_first_successful_post_closure_addition',
            })
        evidence_changed = (
            state['visible_product_ids'] != previous_products
            or state['latest_cart_product_ids'] != previous_cart
            or result_evidence_changed
        )
        new_product_ids = set(product_ids) - previous_products
        cart_changed = cart_product_ids is not None and set(cart_product_ids) != previous_cart
        functional_progress = evidence_changed
        if name in {'search_products', 'filter_products', 'filter_by_range', 'sort_products'}:
            functional_progress = bool(new_product_ids or cart_changed)
        if name in self._PERSIST_LOOKUP_TOOLS:
            state['lookup_signature_counts'][key] = int(state['lookup_signature_counts'].get(key, 0)) + 1
            state['last_lookup_failed'] = bool(failed)
            if functional_progress:
                state['pfl_lookup_stale_run'] = 0
            else:
                state['pfl_lookup_stale_run'] = int(state.get('pfl_lookup_stale_run', 0)) + 1
        if evidence_changed:
            state['evidence_version'] = int(state.get('evidence_version', 0)) + 1
        if (
            not failed
            and state.get('v5_active_closure_subtype') == 'SERIALIZED_REQUIREMENT_CLOSURE'
            and name in self._PERSIST_LOOKUP_TOOLS
        ):
            state['serialized_closure_lookup_count'] = int(
                state.get('serialized_closure_lookup_count', 0)
            ) + 1
        if failed and name in self._PERSIST_LOOKUP_TOOLS:
            state['failed_lookup_counts'][key] = int(state['failed_lookup_counts'].get(key, 0)) + 1
            state['failed_evidence_versions'][key] = int(state.get('evidence_version', 0))
            if self._persist_authoritative_lookup_error(result):
                state['authoritative_failed_lookup_keys'].add(key)
        if failed and name in self._PERSIST_ACTION_TOOLS:
            state['failed_action_keys'].add(key)
            state['failed_action_versions'][key] = int(state.get('evidence_version', 0))
            state['failed_action_tool_counts'][name] = int(
                state['failed_action_tool_counts'].get(name, 0)
            ) + 1
        if not failed and name in self._PERSIST_LOOKUP_TOOLS:
            if evidence_changed:
                state['no_effect_lookup_counts'][key] = 0
            else:
                state['no_effect_lookup_counts'][key] = int(state['no_effect_lookup_counts'].get(key, 0)) + 1
                state['no_effect_evidence_versions'][key] = int(state.get('evidence_version', 0))
        if self._persist_v5_online_enabled() and name in self._PERSIST_LOOKUP_TOOLS:
            no_effect_count = int(state['no_effect_lookup_counts'].get(key, 0))
            failed_count = int(state['failed_lookup_counts'].get(key, 0))
            if no_effect_count == 2 or failed_count == 2:
                if name == 'search_products':
                    suggested_family = 'detail' if state.get('visible_product_ids') else 'search_change_query'
                elif name in {'get_product_details', 'calculate_transport_time'}:
                    suggested_family = 'search_change_query'
                else:
                    suggested_family = 'detail' if state.get('visible_product_ids') else 'search_change_query'
                state['v5_pending_ace'] = {
                    'trigger_family': 'strict_repeated_failure' if failed_count == 2 else 'strict_exact_no_effect',
                    'tool': name,
                    'key': key,
                    'suggested_tool_family': suggested_family,
                    'step': step,
                    'phase': phase,
                }
        state['events'].append({
            'type': 'tool_result',
            'phase': phase,
            'step': step,
            'tool': name,
            'key': key,
            'failed': failed,
            'evidence_version': int(state.get('evidence_version', 0)),
            'new_product_ids': sorted(new_product_ids),
            'cart_changed': cart_changed,
            'result_evidence_changed': result_evidence_changed,
            'functional_progress': functional_progress,
            'authoritative_lookup_error': key in state['authoritative_failed_lookup_keys'],
            'pfl_lookup_stale_run': int(state.get('pfl_lookup_stale_run', 0)),
            'no_effect_count': int(state['no_effect_lookup_counts'].get(key, 0)),
        })
        shadow_event = state.get('v5_shadow_pending', {}).pop(call.get('id'), None)
        if shadow_event is not None:
            shadow_event.update({
                'actual_effect': {
                    'productive': bool(evidence_changed),
                    'failed': bool(failed),
                    'new_product_ids': sorted(set(product_ids) - previous_products),
                    'cart_changed': cart_product_ids is not None and set(cart_product_ids) != previous_cart,
                    'result_evidence_changed': bool(result_evidence_changed),
                },
                'whether_productive': bool(evidence_changed),
            })

    def _persist_proposed_v2_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_v2' in mode or 'proposed_v3' in mode or 'proposed_v4' in mode or 'paper_ready' in mode or 'multicard' in mode

    def _persist_proposed_v3_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_v3' in mode or 'proposed_v4' in mode or 'evidence_compaction' in mode

    def _persist_conservative_v4_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_v4' in mode or 'conservative_context' in mode

    def _persist_v5_shadow_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_v5_shadow' in mode

    def _persist_v5_online_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_v5_minimal' in mode

    def _persist_pfl_plan_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_pfl_plan_minimal' in mode or 'proposed_pfl_plan_score_protective' in mode

    def _persist_pfl_state_guard_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_PFL_STATE_GUARD', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _persist_pfl_state_guard_precard_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_PFL_STATE_GUARD_PRECARD', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _persist_pfl_cb_repeat_safe_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_PFL_CB_REPEAT_SAFE', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _persist_pfl_score_protective_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_pfl_plan_score_protective' in mode

    def _persist_v5_compaction_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        real_overflow_mode = os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_COMPACTION', '').strip().lower()
        return (
            'proposed_v5_minimal_r4' in mode
            or 'proposed_v5_minimal_r5' in mode
            or 'proposed_pfl_plan_minimal_overflow' in mode
            or 'proposed_pfl_plan_score_protective_overflow' in mode
            or real_overflow_mode in {'1', 'true', 'yes', 'on'}
        )

    def _persist_pfl_tiny_compaction_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        real_overflow_tiny = os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_TINY_COMPACTION', '').strip().lower()
        return (
            'proposed_pfl_plan_minimal_overflow_r3' in mode
            or 'proposed_pfl_plan_score_protective_overflow_r3' in mode
            or real_overflow_tiny in {'1', 'true', 'yes', 'on'}
        )

    def _persist_pfl_r4_budget_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_pfl_plan_score_protective_r4' in mode or 'proposed_pfl_plan_score_protective_r5' in mode

    def _persist_pfl_r5_low_completion_model(self, state: Dict[str, Any]) -> Optional[str]:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        if 'proposed_pfl_plan_score_protective_r5' not in mode:
            return None
        raw_cases = os.getenv('SHOPPING_PERSIST_R5_LOW_COMPLETION_CASE_IDS', '').strip()
        allowed = {item.strip().removeprefix('case_') for item in raw_cases.split(',') if item.strip()}
        if str(state.get('sample_id') or '').removeprefix('case_') not in allowed:
            return None
        return os.getenv(
            'SHOPPING_PERSIST_R5_LOW_COMPLETION_MODEL',
            'qwen3-14b-local-persist-pfl-plan-score-protective',
        )

    def _persist_pfl_r4_case_enabled(self, state: Dict[str, Any]) -> bool:
        if not self._persist_pfl_r4_budget_enabled():
            return False
        raw_cases = os.getenv('SHOPPING_PERSIST_R4_CASE_IDS', '').strip()
        if not raw_cases:
            return True
        allowed = {item.strip().removeprefix('case_') for item in raw_cases.split(',') if item.strip()}
        return str(state.get('sample_id') or '').removeprefix('case_') in allowed

    def _persist_pfl_pre_llm_context_guard_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_PFL_PRELLM_CONTEXT_GUARD', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _persist_pfl_pre_llm_max_tokens_override(
        self,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        phase: str,
        step: int,
    ) -> Optional[int]:
        if not self._persist_pfl_pre_llm_context_guard_enabled():
            return None
        char_limit = int(os.getenv('SHOPPING_PERSIST_PFL_PRELLM_CHAR_LIMIT', '62000'))
        current_chars = self._persist_estimate_history_chars(messages)
        if current_chars < char_limit:
            return None
        tight_char_limit = int(os.getenv('SHOPPING_PERSIST_PFL_PRELLM_TIGHT_CHAR_LIMIT', '78000'))
        critical_char_limit = int(os.getenv('SHOPPING_PERSIST_PFL_PRELLM_CRITICAL_CHAR_LIMIT', '84000'))
        if current_chars >= critical_char_limit:
            max_tokens = int(os.getenv('SHOPPING_PERSIST_PFL_PRELLM_CRITICAL_MAX_TOKENS', '128'))
        elif current_chars >= tight_char_limit:
            max_tokens = int(os.getenv('SHOPPING_PERSIST_PFL_PRELLM_TIGHT_MAX_TOKENS', '256'))
        else:
            max_tokens = int(os.getenv('SHOPPING_PERSIST_PFL_PRELLM_MAX_TOKENS', '1024'))
        state['events'].append({
            'type': 'pfl_pre_llm_context_budget_guard',
            'phase': phase,
            'step': step,
            'history_chars': current_chars,
            'char_limit': char_limit,
            'tight_char_limit': tight_char_limit,
            'critical_char_limit': critical_char_limit,
            'max_tokens_override': max_tokens,
            'reason': 'near_model_context_limit_reduce_completion_budget',
        })
        return max_tokens

    def _persist_pfl_r4_search_budget_exceeded(
        self,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        step: int,
    ) -> bool:
        if not self._persist_pfl_r4_case_enabled(state):
            return False
        card = state.get('pfl_card')
        if not isinstance(card, dict):
            return False
        if not state.get('visible_product_ids') and not state.get('latest_cart_product_ids'):
            return False
        score_bucket = str(card.get('score_bucket') or '')
        default_step = 42 if score_bucket == 'partial' else 34
        step_limit = int(os.getenv('SHOPPING_PERSIST_R4_SEARCH_STEP_LIMIT', str(default_step)))
        char_limit = int(os.getenv('SHOPPING_PERSIST_R4_SEARCH_CHAR_LIMIT', '76000'))
        return step >= step_limit or self._persist_estimate_history_chars(messages) >= char_limit

    def _persist_pfl_r4_cart_budget_exceeded(
        self,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        step: int,
    ) -> bool:
        if not self._persist_pfl_r4_case_enabled(state):
            return False
        if step < int(os.getenv('SHOPPING_PERSIST_R4_CART_STEP_LIMIT', '18')):
            return False
        if not state.get('latest_cart_product_ids') and step < int(os.getenv('SHOPPING_PERSIST_R4_CART_EMPTY_STEP_LIMIT', '28')):
            return False
        return True

    def _persist_v5_ace_once_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        return 'proposed_v5_minimal_r5' in mode

    def _persist_pfl_real_overflow_compaction_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_COMPACTION', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _persist_pfl_real_overflow_persist_compaction_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_PERSIST_COMPACTION', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _persist_pfl_real_overflow_completion_shrink_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_COMPLETION_SHRINK', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _persist_pfl_real_overflow_recent_suffix_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_RECENT_SUFFIX', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _persist_pfl_recent_history_suffix(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        char_limit = int(os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_SUFFIX_CHAR_LIMIT', '22000'))
        message_limit = int(os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_SUFFIX_MESSAGE_LIMIT', '12'))
        tool_content_limit = int(os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_SUFFIX_TOOL_CHAR_LIMIT', '2000'))
        first_user = next((message for message in messages if message.get('role') == 'user'), None)
        suffix: List[Dict[str, Any]] = []
        suffix_chars = 0
        for message in reversed(messages):
            if message.get('role') == 'system' or message is first_user:
                continue
            checkpoint_message = dict(message)
            if checkpoint_message.get('role') == 'tool':
                content = str(checkpoint_message.get('content') or '')
                if len(content) > tool_content_limit:
                    checkpoint_message['content'] = json.dumps({
                        'persist_checkpoint': 'full tool result is represented in the State JSON evidence summary',
                        'original_result_chars': len(content),
                    }, ensure_ascii=False, separators=(',', ':'))
            message_chars = self._persist_estimate_history_chars([checkpoint_message])
            completes_leading_tool_group = (
                checkpoint_message.get('role') == 'assistant'
                and suffix
                and suffix[0].get('role') == 'tool'
            )
            if suffix and (
                len(suffix) >= message_limit
                or (suffix_chars + message_chars > char_limit and not completes_leading_tool_group)
            ):
                break
            suffix.insert(0, checkpoint_message)
            suffix_chars += message_chars
        while suffix and suffix[0].get('role') == 'tool':
            suffix.pop(0)
        return suffix

    def _persist_pfl_real_overflow_completion_budget(self, error_text: str) -> Optional[Dict[str, int]]:
        requested_match = re.search(r'requested\D+(\d+)\s+tokens', error_text, re.IGNORECASE)
        context_match = re.search(r'maximum context length\D+(\d+)\s+tokens', error_text, re.IGNORECASE)
        if context_match is None:
            context_match = re.search(r'>\s*(\d+)\b', error_text)
        message_match = re.search(r'(\d+)\s+(?:in\s+)?(?:the\s+)?messages?', error_text, re.IGNORECASE)
        completion_match = re.search(r'(\d+)\s+(?:in\s+)?(?:the\s+)?completion', error_text, re.IGNORECASE)
        if requested_match is None or context_match is None or completion_match is None:
            return None
        requested_tokens = int(requested_match.group(1))
        context_tokens = int(context_match.group(1))
        completion_tokens = int(completion_match.group(1))
        message_tokens = (
            int(message_match.group(1))
            if message_match is not None
            else requested_tokens - completion_tokens
        )
        reserve_tokens = int(os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_COMPLETION_RESERVE', '64'))
        minimum_tokens = int(os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_COMPLETION_MIN', '128'))
        available_tokens = context_tokens - message_tokens - reserve_tokens
        if available_tokens < minimum_tokens or available_tokens >= completion_tokens:
            return None
        return {
            'requested_tokens': requested_tokens,
            'message_tokens': message_tokens,
            'completion_tokens': completion_tokens,
            'context_tokens': context_tokens,
            'reserve_tokens': reserve_tokens,
            'retry_max_tokens': available_tokens,
        }

    def _shopping_transport_overflow_tool_truncation_enabled(self) -> bool:
        mode = os.getenv('SHOPPING_CONTEXT_OVERFLOW_TOOL_TRUNCATION', '').strip().lower()
        return mode in {'1', 'true', 'yes', 'on'}

    def _shopping_transport_trim_old_tool_results(
        self,
        messages: List[Dict[str, Any]],
        error_text: str,
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, int]]]:
        context_match = re.search(r'maximum context length\D+(\d+)\s+tokens', error_text, re.IGNORECASE)
        message_match = re.search(r'(\d+)\s+(?:in\s+)?(?:the\s+)?messages?', error_text, re.IGNORECASE)
        if context_match is None or message_match is None:
            return messages, None
        context_tokens = int(context_match.group(1))
        message_tokens = int(message_match.group(1))
        reserve_tokens = int(os.getenv('SHOPPING_CONTEXT_OVERFLOW_RESERVE_TOKENS', '64'))
        completion_tokens = int(os.getenv('SHOPPING_CONTEXT_OVERFLOW_MIN_COMPLETION_TOKENS', '256'))
        target_message_tokens = context_tokens - reserve_tokens - completion_tokens
        excess_tokens = max(1, message_tokens - target_message_tokens)
        chars_per_token = int(os.getenv('SHOPPING_CONTEXT_OVERFLOW_CHARS_PER_TOKEN', '6'))
        chars_to_remove = max(2048, excess_tokens * chars_per_token)
        compacted = [dict(message) for message in messages]
        removed_chars = 0
        truncated_messages = 0
        marker = '[Earlier tool output truncated by the benchmark transport context policy.]'
        for message in compacted:
            if removed_chars >= chars_to_remove:
                break
            if message.get('role') != 'tool':
                continue
            content = str(message.get('content') or '')
            removable = max(0, len(content) - len(marker))
            if removable <= 0:
                continue
            remove_now = min(removable, chars_to_remove - removed_chars)
            keep_chars = len(content) - remove_now - len(marker)
            message['content'] = content[:max(0, keep_chars)] + marker
            removed_chars += remove_now
            truncated_messages += 1
        if removed_chars <= 0:
            return messages, None
        return compacted, {
            'context_tokens': context_tokens,
            'message_tokens': message_tokens,
            'target_message_tokens': target_message_tokens,
            'completion_tokens': completion_tokens,
            'removed_chars': removed_chars,
            'truncated_tool_messages': truncated_messages,
        }

    def _persist_v5_call_with_overflow_recovery(
        self,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        phase: str,
        step: int,
    ):
        override_model = self._persist_pfl_r5_low_completion_model(state)
        max_tokens_override = self._persist_pfl_pre_llm_max_tokens_override(state, messages, phase, step)
        try:
            return self._call_llm(
                messages=messages,
                tools=tools,
                config_name=override_model,
                max_tokens_override=max_tokens_override,
            )
        except Exception as e:
            error_text = str(e).lower()
            if self._persist_pfl_real_overflow_completion_shrink_enabled() and 'maximum context length' in error_text:
                budget = self._persist_pfl_real_overflow_completion_budget(str(e))
                if budget is not None:
                    state['events'].append({
                        'type': 'pfl_real_overflow_completion_shrink',
                        'phase': phase,
                        'step': step,
                        'reason': 'actual_context_overflow_completion_only_recovery',
                        **budget,
                    })
                    try:
                        return self._call_llm(
                            messages=messages,
                            tools=tools,
                            config_name=override_model,
                            max_tokens_override=budget['retry_max_tokens'],
                        )
                    except Exception as retry_error:
                        if 'maximum context length' not in str(retry_error).lower():
                            raise
                        e = retry_error
                        error_text = str(retry_error).lower()
            if self._shopping_transport_overflow_tool_truncation_enabled() and 'maximum context length' in error_text:
                transport_messages = messages
                max_attempts = int(os.getenv('SHOPPING_CONTEXT_OVERFLOW_TOOL_TRUNCATION_ATTEMPTS', '3'))
                for transport_attempt in range(1, max_attempts + 1):
                    compacted, transport = self._shopping_transport_trim_old_tool_results(
                        transport_messages,
                        str(e),
                    )
                    if transport is None:
                        break
                    state['events'].append({
                        'type': 'transport_context_tool_result_truncation',
                        'phase': phase,
                        'step': step,
                        'attempt': transport_attempt,
                        'reason': 'request_messages_exceed_model_context',
                        **transport,
                    })
                    try:
                        return self._call_llm(
                            messages=compacted,
                            tools=tools,
                            config_name=override_model,
                            max_tokens_override=transport['completion_tokens'],
                        )
                    except Exception as transport_error:
                        if 'maximum context length' not in str(transport_error).lower():
                            raise
                        transport_messages = compacted
                        e = transport_error
                        error_text = str(transport_error).lower()
            count = int(state.get('v5_compaction_counts', {}).get(phase, 0))
            mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
            if self._persist_pfl_real_overflow_compaction_enabled():
                max_compactions = int(os.getenv('SHOPPING_PERSIST_PFL_REAL_OVERFLOW_MAX_COMPACTIONS', '8'))
            else:
                max_compactions = 12 if 'proposed_pfl_plan_minimal_overflow_r3' in mode else 2 if self._persist_v5_ace_once_enabled() else 1
            if (
                not self._persist_v5_compaction_enabled()
                or 'maximum context length' not in error_text
                or count >= max_compactions
            ):
                raise
            compacted = self._persist_v5_state_compaction(state, messages, phase, step)
            compacted_max_tokens_override = self._persist_pfl_pre_llm_max_tokens_override(state, compacted, phase, step)
            response = self._call_llm(
                messages=compacted,
                tools=tools,
                config_name=override_model,
                max_tokens_override=compacted_max_tokens_override,
            )
            if self._persist_pfl_real_overflow_persist_compaction_enabled():
                before_len = len(messages)
                messages[:] = compacted
                state['events'].append({
                    'type': 'v5_state_compaction_persisted',
                    'phase': phase,
                    'step': step,
                    'reason': 'actual_model_context_overflow_persist_compacted_llm_history',
                    'before_messages': before_len,
                    'after_messages': len(messages),
                    'after_chars': self._persist_estimate_history_chars(messages),
                })
            return response

    def _persist_v5_state_compaction(
        self,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        phase: str,
        step: int,
    ) -> List[Dict[str, Any]]:
        ledger = state.get('v5_requirement_ledger') or []
        evidence = self._persist_build_evidence_summary(messages)
        if self._persist_pfl_tiny_compaction_enabled():
            evidence = dict(evidence)
            evidence['candidate_product_ids'] = evidence.get('candidate_product_ids', [])[:8]
            evidence['product_evidence'] = evidence.get('product_evidence', [])[:3]
            evidence['recent_tool_calls'] = evidence.get('recent_tool_calls', [])[-4:]
            evidence['repeated_errors'] = evidence.get('repeated_errors', [])[:2]
            ledger = ledger[:4]
        system_message = next((message for message in messages if message.get('role') == 'system'), None)
        user_messages = [message for message in messages if message.get('role') == 'user']
        compacted: List[Dict[str, Any]] = []
        if system_message:
            compacted.append(system_message)
        compacted.append({
            'role': 'system',
            'content': (
                '[PERSIST-v5 overflow recovery] The original history exceeded the real model context limit. '
                'Continue from this deterministic state summary without restarting broad retrieval. '
                'Use only real product/coupon IDs in the summary. Resolve remaining requirements, repair the cart if needed, '
                'and finish normally when the cart/evidence is sufficient. '
                + (
                    'A bounded recent assistant/tool suffix follows the original user request; treat it as the latest '
                    'execution checkpoint and do not repeat work already completed there. '
                    if self._persist_pfl_real_overflow_recent_suffix_enabled()
                    else ''
                )
                + 'State JSON: ' +
                json.dumps({'requirements': ledger, 'evidence': evidence}, ensure_ascii=False, separators=(',', ':'))
            ),
        })
        if user_messages:
            compacted.append(user_messages[0])
        suffix: List[Dict[str, Any]] = []
        if self._persist_pfl_real_overflow_recent_suffix_enabled():
            suffix = self._persist_pfl_recent_history_suffix(messages)
            compacted.extend(suffix)
        elif phase == 'cart_check' and len(user_messages) > 1:
            compacted.append(user_messages[-1])
        state['v5_compaction_counts'][phase] = int(state['v5_compaction_counts'].get(phase, 0)) + 1
        state['events'].append({
            'type': 'v5_state_compaction',
            'phase': phase,
            'step': step,
            'reason': 'actual_model_context_overflow',
            'before_chars': self._persist_estimate_history_chars(messages),
            'after_chars': self._persist_estimate_history_chars(compacted),
            'candidate_ids': len(evidence.get('candidate_product_ids', [])),
            'product_evidence': len(evidence.get('product_evidence', [])),
            'recent_suffix_messages': len(suffix),
            'recent_suffix_chars': self._persist_estimate_history_chars(suffix),
        })
        return compacted

    def _persist_load_v5_ledger(self, state: Dict[str, Any], messages_file: Path) -> None:
        if not self._persist_v5_online_enabled() and not self._persist_pfl_plan_enabled():
            return
        if self._persist_ablation_mode() == 'no_requirement_ledger':
            state['events'].append({
                'type': 'pfl_ablation_component_disabled',
                'component': 'requirement_ledger',
            })
            return
        ledger_path = messages_file.with_name('requirement_ledger.json')
        try:
            with open(ledger_path, encoding='utf-8') as f:
                payload = json.load(f)
            requirements = payload.get('requirements')
            if not isinstance(requirements, list) or not requirements:
                raise ValueError('requirements must be a non-empty list')
            compact = []
            for index, requirement in enumerate(requirements, 1):
                if not isinstance(requirement, dict):
                    continue
                compact.append({
                    'requirement_id': str(requirement.get('requirement_id') or f'R{index}'),
                    'product_description': str(requirement.get('product_description') or ''),
                    'quantity': max(1, int(requirement.get('quantity', 1))),
                    'hard_constraints': [str(value) for value in requirement.get('hard_constraints', [])],
                })
            if not compact:
                raise ValueError('ledger contains no valid requirements')
            state['v5_requirement_ledger'] = compact
            self._persist_refresh_requirement_state(state)
            state['events'].append({
                'type': 'v5_state_persistence',
                'reason': 'requirement_ledger_loaded',
                'requirement_count': len(compact),
                'ledger_path': str(ledger_path),
            })
        except Exception as e:
            state['events'].append({
                'type': 'v5_state_persistence_error',
                'reason': 'requirement_ledger_load_failed',
                'ledger_path': str(ledger_path),
                'error': f'{type(e).__name__}: {e}',
            })

    def _persist_load_pfl_card(self, state: Dict[str, Any], messages_file: Path) -> None:
        if not self._persist_pfl_plan_enabled():
            return
        card_path = messages_file.with_name('pfl_card.json')
        try:
            with open(card_path, encoding='utf-8') as f:
                card = json.load(f)
            if not isinstance(card, dict) or not card.get('recommended_card'):
                raise ValueError('pfl_card must contain recommended_card')
            if card.get('selected') is False:
                raise ValueError('pfl_card is not selected')
            state['pfl_card'] = card
            if self._persist_pfl_score_protective_enabled() and str(card.get('score_bucket')) != 'zero':
                state['pfl_card_shadow_only'] = True
            state['events'].append({
                'type': 'pfl_card_loaded',
                'card_path': str(card_path),
                'recommended_card': card.get('recommended_card'),
                'pfl_family': card.get('pfl_family'),
                'first_trigger_step': ((card.get('first_trigger') or {}).get('step')),
                'score_bucket': card.get('score_bucket'),
                'shadow_only': bool(state.get('pfl_card_shadow_only')),
            })
        except Exception as e:
            state['events'].append({
                'type': 'pfl_card_load_error',
                'card_path': str(card_path),
                'error': f'{type(e).__name__}: {e}',
            })

    def _persist_pfl_target_phase(self, card: Dict[str, Any]) -> str:
        target_phase = str(card.get('target_phase') or '').strip()
        if target_phase in {'search', 'cart_check', 'any'}:
            return target_phase
        return 'cart_check' if card.get('recommended_card') == 'cart_reconciliation_card' else 'search'

    def _persist_pfl_card_message(self, state: Dict[str, Any], card: Dict[str, Any]) -> str:
        ledger = state.get('v5_requirement_ledger') or []
        requirement_state = state.get('v5_requirement_state') or {}
        first_trigger = card.get('first_trigger') or {}
        active_closure_subtype = state.get('v5_active_closure_subtype') or card.get('closure_subtype')
        prompt_templates = card.get('prompt_templates') or {}
        prompt_template = prompt_templates.get(active_closure_subtype, card.get('prompt_template', ''))
        card_ledger = ledger
        card_requirement_state = requirement_state
        allowed_range_keys: set[str] = set()
        budget_window: tuple[float | None, float | None] = (None, None)
        if active_closure_subtype:
            unresolved_rows = [
                requirement
                for requirement in requirement_state.get('requirements', [])
                if not requirement.get('solved')
            ]
            unresolved_ids = {
                str(requirement.get('requirement_id'))
                for requirement in unresolved_rows
                if requirement.get('requirement_id')
            }
            card_ledger = [
                requirement
                for requirement in ledger
                if str(requirement.get('requirement_id')) in unresolved_ids
            ]
            card_requirement_state = {
                key: value
                for key, value in requirement_state.items()
                if key != 'requirements'
            }
            card_requirement_state['requirements'] = unresolved_rows
            for requirement in card_ledger:
                for _, path, _, _ in numeric_constraints(text_for_requirement(requirement)):
                    if isinstance(path, tuple):
                        allowed_range_keys.add('.'.join(path))
            if len(card_ledger) == 1:
                budget_window = remaining_unit_price_window(
                    state.get('user_query', ''),
                    state.get('latest_cart_total'),
                    int(card_ledger[0].get('quantity', 1)),
                )
                if budget_window != (None, None):
                    allowed_range_keys.add('price')
        payload = {
            'recommended_card': card.get('recommended_card'),
            'pfl_family': card.get('pfl_family'),
            'card_selection_reason': card.get('card_selection_reason'),
            'first_trigger': {
                'step': first_trigger.get('step'),
                'missing_requirement_count': first_trigger.get('missing_requirement_count'),
                'supported_candidate_count': first_trigger.get('supported_candidate_count'),
                'visible_candidate_count': first_trigger.get('visible_candidate_count'),
                'cart_size': first_trigger.get('cart_size'),
                'stale_run': first_trigger.get('stale_run'),
            },
            'requirements': card_ledger,
            'online_requirement_state': card_requirement_state,
            'closure_subtype': active_closure_subtype,
            'recoverability_subtype': state.get('v5_trigger_closure_subtype'),
            'recovery_contract': {
                'allowed_range_keys': sorted(allowed_range_keys),
                'global_filter_required_when_no_supported_ids': bool(
                    active_closure_subtype
                    and unresolved_rows
                    and not unresolved_rows[0].get('supported_ids')
                    and allowed_range_keys
                ),
                'first_global_filter_omits_product_ids': True,
                'maximum_candidate_details': 3,
                'maximum_cart_additions': 1,
                'budget_closure': {
                    'current_cart_total': state.get('latest_cart_total'),
                    'required_unit_price_min': budget_window[0],
                    'required_unit_price_max': budget_window[1],
                    'modify_existing_cart_items': False,
                },
            },
        }
        return (
            '[PERSIST-ACE-Plan] Apply this one-time typed recovery card. '
            'Do not stop merely because this card appears. Do not invent product IDs. '
            'Do not force cart completion; use only real tool-returned product/coupon IDs. '
            'Do not delete or replace supported cart items unless visible evidence shows they violate a requirement. '
            'If recovery_contract.budget_closure contains a price window, apply it as an online-visible hard filter before selecting a product. '
            f"{prompt_template} "
            'Card JSON: ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        )

    def _persist_premature_finalize_message(
        self,
        state: Dict[str, Any],
        phase: str,
        step: int,
    ) -> Optional[str]:
        if not state.get('enabled') or not self._persist_pfl_plan_enabled():
            return None
        card = state.get('pfl_card')
        if not isinstance(card, dict) or state.get('pfl_card_prompted'):
            return None
        if state.get('pfl_card_shadow_only'):
            return None
        allowed_subtypes = set(card.get('closure_subtypes') or [])
        if card.get('closure_subtype'):
            allowed_subtypes.add(str(card.get('closure_subtype')))
        supported_finalize = 'SUPPORTED_PREMATURE_FINALIZE' in allowed_subtypes
        serialized_closure = 'SERIALIZED_REQUIREMENT_CLOSURE' in allowed_subtypes
        if (
            'PREMATURE_FINALIZE' not in allowed_subtypes
            and not supported_finalize
            and not serialized_closure
        ):
            return None
        requirement_state = state.get('v5_requirement_state') or {}
        solved_count = int(requirement_state.get('solved_count', 0))
        missing_count = int(requirement_state.get('missing_requirement_count', 0))
        missing_rows = [
            requirement
            for requirement in requirement_state.get('requirements', [])
            if not requirement.get('solved')
        ]
        cart_item_count = len(state.get('latest_cart_product_ids') or set())
        missing_requirement_definition = next(
            (
                requirement
                for requirement in state.get('v5_requirement_ledger', [])
                if str(requirement.get('requirement_id'))
                == str(missing_rows[0].get('requirement_id'))
            ),
            None,
        )
        serialized_numeric_constraints = (
            numeric_constraints(text_for_requirement(missing_requirement_definition))
            if isinstance(missing_requirement_definition, dict)
            else []
        )
        if (
            solved_count < 1
            or missing_count != 1
            or len(missing_rows) != 1
            or (bool(missing_rows[0].get('supported_ids')) != supported_finalize)
            or cart_item_count != solved_count
            or (serialized_closure and len(serialized_numeric_constraints) < 2)
        ):
            return None
        active_subtype = (
            'SUPPORTED_PREMATURE_FINALIZE'
            if supported_finalize
            else 'SERIALIZED_REQUIREMENT_CLOSURE'
            if serialized_closure
            else 'PREMATURE_FINALIZE'
        )
        state['v5_active_closure_subtype'] = active_subtype
        state['pfl_card_prompted'] = True
        gate_state = {
            'phase': phase,
            'step': step,
            'action': 'assistant_finalize_without_tool',
            'solved_requirement_count': solved_count,
            'missing_requirement_count': missing_count,
            'missing_requirement_id': missing_rows[0].get('requirement_id'),
            'missing_supported_ids': list(missing_rows[0].get('supported_ids') or []),
            'closure_subtype': active_subtype,
            'serialized_numeric_constraint_count': len(serialized_numeric_constraints),
            'cart_item_count': cart_item_count,
            'cart_size_equals_solved_count': cart_item_count == solved_count,
        }
        state['events'].append({
            'type': 'pfl_online_intervention',
            'phase': phase,
            'step': step,
            'recommended_card': card.get('recommended_card'),
            'pfl_family': card.get('pfl_family'),
            'decision': 'one_time_premature_finalize_recovery_without_tool_injection',
            'target_phase': 'cart_check',
            'trigger_mode': 'assistant_finalize_without_tool',
            'handoff_gate_state': gate_state,
        })
        return self._persist_pfl_card_message(state, card)

    def _persist_extract_result_evidence(self, tool_name: str, result: str) -> tuple[set[str], Optional[set[str]]]:
        try:
            parsed = json.loads(result or '{}')
        except Exception:
            return set(), None
        product_ids: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                product_id = value.get('product_id')
                if isinstance(product_id, str) and product_id:
                    product_ids.add(product_id)
                for key in ('product_ids', 'filtered_products_ids'):
                    ids = value.get(key)
                    if isinstance(ids, list):
                        product_ids.update(str(item) for item in ids if item)
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(parsed)
        cart_product_ids: Optional[set[str]] = None
        if tool_name in self._PERSIST_ACTION_TOOLS or tool_name == 'get_cart_info':
            if isinstance(parsed, dict) and isinstance(parsed.get('items'), list):
                cart_product_ids = {
                    str(item.get('product_id'))
                    for item in parsed['items']
                    if isinstance(item, dict) and item.get('product_id')
                }
        return product_ids, cart_product_ids

    def _persist_compact_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        fields = (
            'product_id', 'name', 'price', 'brand', 'color', 'size', 'stock_quantity',
            'suitable_season', 'target_demographic', 'sales_volume', 'rating',
            'shipping_info', 'applicable_coupons',
        )
        return {field: product[field] for field in fields if field in product}

    def _persist_build_evidence_summary(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        products: Dict[str, Dict[str, Any]] = {}
        candidate_ids: set[str] = set()
        latest_cart: Optional[Dict[str, Any]] = None
        errors: Counter = Counter()
        recent_calls: List[Dict[str, Any]] = []
        for message in messages:
            if message.get('role') == 'assistant' and message.get('tool_calls'):
                for call in message['tool_calls']:
                    function = call.get('function', call) if isinstance(call, dict) else {}
                    if not isinstance(function, dict) or not function.get('name'):
                        continue
                    recent_calls.append({
                        'name': function.get('name'),
                        'arguments': self._persist_normalize_arguments(function.get('arguments', '')),
                    })
            if message.get('role') != 'tool':
                continue
            content = str(message.get('content') or '')
            try:
                parsed = json.loads(content)
            except Exception:
                if content:
                    errors[content[:240]] += 1
                continue
            if isinstance(parsed, dict):
                for key in ('product_ids', 'filtered_products_ids'):
                    ids = parsed.get(key)
                    if isinstance(ids, list):
                        candidate_ids.update(str(item) for item in ids if item)
                if isinstance(parsed.get('products'), list):
                    for product in parsed['products']:
                        if not isinstance(product, dict) or not product.get('product_id'):
                            continue
                        product_id = str(product['product_id'])
                        candidate_ids.add(product_id)
                        products[product_id] = self._persist_compact_product(product)
                if isinstance(parsed.get('items'), list) and 'summary' in parsed:
                    latest_cart = {
                        'items': [self._persist_compact_product(item) for item in parsed['items'] if isinstance(item, dict)],
                        'used_coupons': parsed.get('used_coupons', []),
                        'summary': parsed.get('summary', {}),
                    }
                if parsed.get('error'):
                    errors[str(parsed['error'])[:240]] += 1
        selected_ids = list(products)[:40]
        remaining_ids = sorted(candidate_ids - set(selected_ids))
        return {
            'candidate_product_ids': (selected_ids + remaining_ids)[:80],
            'product_evidence': [products[product_id] for product_id in selected_ids],
            'latest_cart': latest_cart or {'items': [], 'used_coupons': [], 'summary': {}},
            'repeated_errors': [{'message': message, 'count': count} for message, count in errors.most_common(8)],
            'recent_tool_calls': recent_calls[-24:],
        }

    def _persist_pfl_handoff_gate(
        self,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        card: Dict[str, Any],
        phase: str,
        step: int,
    ) -> tuple[bool, Dict[str, Any], str | None]:
        trigger_mode = card.get('trigger_mode')
        if trigger_mode in {
            'online_evidence_stagnation',
            'online_ready_to_detail_stagnation',
            'online_partial_coverage_stagnation',
            'online_requirement_closure',
        }:
            min_step = int(card.get('online_gate_min_step') or 1)
            min_global_step = int(card.get('online_gate_min_global_step') or min_step)
            min_stale_run = int(card.get('online_gate_min_stale_run') or 4)
            min_visible_candidates = int(card.get('online_gate_min_visible_candidates') or 1)
            max_cart_items = int(card.get('online_gate_max_cart_items') or 0)
            ledger = state.get('v5_requirement_ledger') or []
            visible_candidate_count = len(state.get('visible_product_ids') or [])
            cart_item_count = len(state.get('latest_cart_product_ids') or [])
            stale_run = int(state.get('pfl_lookup_stale_run', 0))
            last_lookup_key = state.get('last_executed_call_key')
            last_lookup_tool = state.get('last_executed_call_tool')
            last_lookup_signature_count = int(
                (state.get('lookup_signature_counts') or {}).get(last_lookup_key, 0)
            )
            failed_action_tool_counts = state.get('failed_action_tool_counts') or {}
            coupon_failure_count = sum(
                int(failed_action_tool_counts.get(name, 0))
                for name in ('add_coupon_to_cart', 'delete_coupon_from_cart')
            )
            detail_product_count = len(state.get('detail_product_ids') or [])
            requirement_state = state.get('v5_requirement_state') or {}
            requirement_delta = state.get('v5_requirement_last_delta') or {}
            requirement_satisfaction_stale_run = int(
                state.get('v5_requirement_satisfaction_stale_run', 0)
            )
            recent_requirement_satisfaction_churn = int(
                state.get('v5_requirement_recent_satisfaction_churn', 0)
            )
            gate_state = {
                'trigger_mode': trigger_mode,
                'phase': phase,
                'step': step,
                'min_step': min_step,
                'global_llm_step': int(state.get('pfl_global_llm_step', 0)),
                'min_global_step': min_global_step,
                'requirement_count': len(ledger),
                'visible_candidate_count': visible_candidate_count,
                'min_visible_candidates': min_visible_candidates,
                'cart_item_count': cart_item_count,
                'max_cart_items': max_cart_items,
                'lookup_stale_run': stale_run,
                'min_stale_run': min_stale_run,
                'last_lookup_tool': last_lookup_tool,
                'last_lookup_signature_count': last_lookup_signature_count,
                'last_lookup_failed': state.get('last_lookup_failed'),
                'coupon_failure_count': coupon_failure_count,
                'detail_product_count': detail_product_count,
                'requirement_state_hash': requirement_state.get('ledger_hash'),
                'solved_requirement_count': int(requirement_state.get('solved_count', 0)),
                'missing_requirement_count': int(requirement_state.get('missing_requirement_count', 0)),
                'requirement_delta': requirement_delta,
                'requirement_satisfaction_stale_run': requirement_satisfaction_stale_run,
                'recent_requirement_satisfaction_churn': recent_requirement_satisfaction_churn,
            }
            if trigger_mode == 'online_requirement_closure':
                if phase not in {'search', 'cart_check'}:
                    return False, gate_state, 'not_requirement_closure_phase'
            elif phase != 'search' and trigger_mode != 'online_partial_coverage_stagnation':
                return False, gate_state, 'not_search_phase'
            if trigger_mode == 'online_partial_coverage_stagnation':
                if phase not in {'search', 'cart_check'}:
                    return False, gate_state, 'not_coverage_recovery_phase'
                if int(state.get('pfl_global_llm_step', 0)) < min_global_step:
                    return False, gate_state, 'before_online_gate_min_global_step'
            elif step < min_step:
                return False, gate_state, 'before_online_gate_min_step'
            if not ledger and self._persist_ablation_mode() != 'no_requirement_ledger':
                return False, gate_state, 'missing_requirement_ledger'
            if trigger_mode == 'online_requirement_closure':
                if not requirement_state:
                    return False, gate_state, 'missing_shared_requirement_state'
                if int(requirement_state.get('solved_count', 0)) < 1:
                    return False, gate_state, 'no_solved_requirement'
                if int(requirement_state.get('missing_requirement_count', 0)) != 1:
                    return False, gate_state, 'not_single_missing_requirement'
                if (
                    int(requirement_delta.get('solved_count', 0)) > 0
                    or int(requirement_delta.get('verified_count', 0)) > 0
                    or int(requirement_delta.get('supported_candidate_count', 0)) > 0
                    or int(requirement_delta.get('missing_requirement_count', 0)) < 0
                ):
                    return False, gate_state, 'current_action_made_requirement_progress'
                if recent_requirement_satisfaction_churn != 0:
                    return False, gate_state, 'recent_requirement_progress_is_productive'
                allowed_subtypes = set(card.get('closure_subtypes') or [])
                if not allowed_subtypes and card.get('closure_subtype'):
                    allowed_subtypes.add(str(card.get('closure_subtype')))
                subtype = None
                if last_lookup_key in (state.get('authoritative_failed_lookup_keys') or set()):
                    subtype = 'AUTHORITATIVE_CLOSURE_BLOCKER'
                elif (
                    'COUPON_OSCILLATION_MISSING_PREREQUISITE' in allowed_subtypes
                    and phase == 'cart_check'
                    and last_lookup_tool in {'add_coupon_to_cart', 'delete_coupon_from_cart'}
                    and coupon_failure_count >= 2
                    and cart_item_count == int(requirement_state.get('solved_count', 0))
                    and cart_item_count <= len(ledger)
                    and requirement_satisfaction_stale_run >= 2
                ):
                    subtype = 'COUPON_OSCILLATION_MISSING_PREREQUISITE'
                elif (
                    'REPEATED_CART_CHECK_MISSING_EVIDENCE' in allowed_subtypes
                    and phase == 'cart_check'
                    and last_lookup_tool == 'get_cart_info'
                    and last_lookup_signature_count >= 2
                    and cart_item_count >= int(requirement_state.get('solved_count', 0))
                    and cart_item_count <= len(ledger)
                    and requirement_satisfaction_stale_run >= 2
                ):
                    subtype = 'REPEATED_CART_CHECK_MISSING_EVIDENCE'
                elif (
                    phase == 'cart_check'
                    and last_lookup_tool == 'get_cart_info'
                    and last_lookup_signature_count >= 2
                    and cart_item_count >= 1
                    and requirement_satisfaction_stale_run >= 6
                ):
                    subtype = 'REPEATED_PREMATURE_CART_CHECK'
                elif (
                    'MISSING_DETAIL_GROUNDING' in allowed_subtypes
                    and
                    last_lookup_tool in {
                        'search_products',
                        'filter_by_brand',
                        'filter_by_color',
                        'filter_by_size',
                        'filter_by_range',
                        'sort_products',
                    }
                    and last_lookup_signature_count >= 2
                ):
                    subtype = 'MISSING_DETAIL_GROUNDING'
                elif last_lookup_tool == 'search_products' and last_lookup_signature_count >= 2:
                    subtype = 'MISSING_REQUIREMENT_QUERY_COLLAPSE'
                gate_state['closure_subtype'] = subtype
                gate_state['allowed_closure_subtypes'] = sorted(allowed_subtypes)
                if subtype not in allowed_subtypes:
                    return False, gate_state, 'current_action_not_allowed_closure_subtype'
                if subtype == 'MISSING_DETAIL_GROUNDING':
                    if visible_candidate_count < 1:
                        return False, gate_state, 'no_prior_grounded_product_context'
                    if last_lookup_signature_count < 2:
                        return False, gate_state, 'detail_grounding_retrieval_not_repeated'
                elif subtype == 'MISSING_REQUIREMENT_QUERY_COLLAPSE':
                    missing_rows = [
                        requirement
                        for requirement in requirement_state.get('requirements', [])
                        if not requirement.get('solved')
                    ]
                    if len(missing_rows) != 1 or missing_rows[0].get('supported_ids'):
                        return False, gate_state, 'missing_requirement_already_has_supported_product'
                    if last_lookup_tool != 'search_products':
                        return False, gate_state, 'last_action_not_collapsed_search_query'
                    if state.get('last_lookup_failed') is not False:
                        return False, gate_state, 'last_collapsed_search_failed'
                    if last_lookup_signature_count < 2:
                        return False, gate_state, 'collapsed_search_query_not_repeated'
                    if visible_candidate_count < 1:
                        return False, gate_state, 'no_prior_grounded_product_context'
                elif subtype == 'AUTHORITATIVE_CLOSURE_BLOCKER':
                    if last_lookup_key not in (state.get('authoritative_failed_lookup_keys') or set()):
                        return False, gate_state, 'last_action_not_authoritative_lookup_error'
                elif subtype == 'REPEATED_PREMATURE_CART_CHECK':
                    missing_rows = [
                        requirement
                        for requirement in requirement_state.get('requirements', [])
                        if not requirement.get('solved')
                    ]
                    if len(missing_rows) != 1 or missing_rows[0].get('supported_ids'):
                        return False, gate_state, 'missing_requirement_already_has_supported_product'
                elif subtype in {
                    'REPEATED_CART_CHECK_MISSING_EVIDENCE',
                    'COUPON_OSCILLATION_MISSING_PREREQUISITE',
                }:
                    missing_rows = [
                        requirement
                        for requirement in requirement_state.get('requirements', [])
                        if not requirement.get('solved')
                    ]
                    if len(missing_rows) != 1 or missing_rows[0].get('supported_ids'):
                        return False, gate_state, 'missing_requirement_already_has_supported_product'
                    if cart_item_count > len(ledger):
                        return False, gate_state, 'cart_exceeds_requirement_scope'
                else:
                    return False, gate_state, 'unknown_requirement_closure_subtype'
                state['v5_trigger_closure_subtype'] = subtype
                state['v5_active_closure_subtype'] = (
                    'SERIALIZED_REQUIREMENT_CLOSURE'
                    if subtype in {
                        'REPEATED_CART_CHECK_MISSING_EVIDENCE',
                        'COUPON_OSCILLATION_MISSING_PREREQUISITE',
                    }
                    else subtype
                )
                gate_state['execution_subtype'] = state['v5_active_closure_subtype']
                return True, gate_state, None
            if visible_candidate_count < min_visible_candidates:
                return False, gate_state, 'insufficient_visible_candidates'
            if cart_item_count > max_cart_items:
                return False, gate_state, 'cart_state_not_bounded'
            if stale_run < min_stale_run:
                return False, gate_state, 'insufficient_lookup_stagnation'
            if trigger_mode == 'online_ready_to_detail_stagnation':
                allowed_filter_tools = {
                    'filter_by_brand',
                    'filter_by_color',
                    'filter_by_size',
                    'filter_by_range',
                    'sort_products',
                }
                min_repeated_signature_count = int(
                    card.get('online_gate_min_repeated_signature_count') or 2
                )
                gate_state['min_repeated_signature_count'] = min_repeated_signature_count
                if last_lookup_tool not in allowed_filter_tools:
                    return False, gate_state, 'last_lookup_not_detail_filter'
                if state.get('last_lookup_failed') is not False:
                    return False, gate_state, 'last_lookup_had_tool_error'
                if last_lookup_signature_count < min_repeated_signature_count:
                    return False, gate_state, 'lookup_signature_not_repeated'
                if detail_product_count > 0:
                    return False, gate_state, 'product_details_already_available'
            if trigger_mode == 'online_partial_coverage_stagnation':
                allowed_refocus_tools = {
                    'search_products',
                    'filter_by_brand',
                    'filter_by_color',
                    'filter_by_size',
                    'filter_by_range',
                    'sort_products',
                    'get_cart_info',
                }
                requirement_count = len(ledger)
                gate_state['partial_coverage_requirement_count'] = requirement_count
                if requirement_count < 3:
                    return False, gate_state, 'insufficient_multi_requirement_scope'
                if cart_item_count <= 0 or cart_item_count >= requirement_count:
                    return False, gate_state, 'cart_not_partially_covered'
                if detail_product_count <= 0:
                    return False, gate_state, 'missing_grounded_product_details'
                if last_lookup_tool not in allowed_refocus_tools:
                    return False, gate_state, 'last_lookup_not_read_only_refocus'
                if state.get('last_lookup_failed') is not False:
                    return False, gate_state, 'last_lookup_had_tool_error'
            return True, gate_state, None
        if trigger_mode != 'phase_handoff_unresolved_constraints':
            return True, {}, None
        if phase != 'cart_check':
            return False, {}, 'not_cart_check_phase'
        min_step = int(card.get('handoff_min_step') or 1)
        if step < min_step:
            return False, {'handoff_min_step': min_step}, 'before_handoff_min_step'
        max_message_chars = int(os.getenv('SHOPPING_PERSIST_PFL_HANDOFF_MAX_MESSAGE_CHARS', '70000'))
        message_chars = sum(len(str(message.get('content') or '')) for message in messages)
        if message_chars > max_message_chars:
            return False, {
                'trigger_mode': card.get('trigger_mode'),
                'phase': phase,
                'step': step,
                'message_chars': message_chars,
                'max_message_chars': max_message_chars,
            }, 'handoff_context_too_large'
        ledger = state.get('v5_requirement_ledger') or []
        evidence = self._persist_build_evidence_summary(messages)
        last_assistant_text = ''
        for message in reversed(messages):
            if message.get('role') == 'assistant' and not message.get('tool_calls'):
                last_assistant_text = str(message.get('content') or '')
                break
        lowered_last_assistant = last_assistant_text.lower()
        terminal_failure_markers = (
            "couldn't find",
            "could not find",
            "can't find",
            "cannot find",
            "no valid",
            "no items",
            "no combinations",
            "unable to find",
            "not find any",
            "no products",
        )
        has_terminal_search_failure = any(marker in lowered_last_assistant for marker in terminal_failure_markers)
        latest_cart = evidence.get('latest_cart') or {}
        cart_items = latest_cart.get('items') if isinstance(latest_cart, dict) else []
        if not isinstance(cart_items, list):
            cart_items = []
        required_quantity = 0
        for requirement in ledger:
            if not isinstance(requirement, dict):
                continue
            try:
                required_quantity += max(1, int(requirement.get('quantity', 1)))
            except Exception:
                required_quantity += 1
        candidate_count = len(evidence.get('candidate_product_ids') or [])
        product_evidence_count = len(evidence.get('product_evidence') or [])
        recent_lookup_count = sum(
            1
            for call in evidence.get('recent_tool_calls', [])
            if isinstance(call, dict) and call.get('name') in self._PERSIST_LOOKUP_TOOLS
        )
        require_empty_cart = bool(card.get('handoff_require_empty_cart', True))
        gate_state = {
            'trigger_mode': card.get('trigger_mode'),
            'phase': phase,
            'step': step,
            'requirement_count': len(ledger),
            'required_quantity': required_quantity,
            'cart_item_count': len(cart_items),
            'candidate_count': candidate_count,
            'product_evidence_count': product_evidence_count,
            'recent_lookup_count': recent_lookup_count,
            'require_empty_cart': require_empty_cart,
            'require_search_terminal_failure': bool(card.get('handoff_require_search_terminal_failure', False)),
            'has_search_terminal_failure': has_terminal_search_failure,
            'last_assistant_excerpt': last_assistant_text[:500],
        }
        if card.get('handoff_require_search_terminal_failure') and not has_terminal_search_failure:
            return False, gate_state, 'search_phase_not_terminal_failure'
        if not ledger:
            return False, gate_state, 'missing_requirement_ledger'
        if required_quantity <= 0:
            return False, gate_state, 'empty_required_quantity'
        if candidate_count <= 0 and product_evidence_count <= 0:
            return False, gate_state, 'no_visible_candidate_or_product_evidence'
        if recent_lookup_count <= 0:
            return False, gate_state, 'no_prior_lookup_evidence'
        if require_empty_cart and len(cart_items) > 0:
            return False, gate_state, 'cart_not_empty'
        if len(cart_items) >= required_quantity:
            return False, gate_state, 'cart_not_missing_required_items'
        return True, gate_state, None

    def _persist_messages_for_llm(self, state: Dict[str, Any], messages: List[Dict[str, Any]], phase: str, step: int) -> List[Dict[str, Any]]:
        if state.get('enabled') and self._persist_pfl_plan_enabled():
            state['pfl_global_llm_step'] = int(state.get('pfl_global_llm_step', 0)) + 1
            card = state.get('pfl_card')
            if not isinstance(card, dict) or state.get('pfl_card_prompted'):
                return messages
            if state.get('pfl_card_shadow_only'):
                if not state.get('pfl_card_shadow_logged'):
                    state['pfl_card_shadow_logged'] = True
                    state['events'].append({
                        'type': 'pfl_online_intervention_skipped',
                        'phase': phase,
                        'step': step,
                        'reason': 'score_protective_partial_score_shadow_only',
                        'recommended_card': card.get('recommended_card'),
                        'pfl_family': card.get('pfl_family'),
                        'score_bucket': card.get('score_bucket'),
                    })
                return messages
            target_phase = self._persist_pfl_target_phase(card)
            trigger_step = int(((card.get('first_trigger') or {}).get('step')) or 1)
            if card.get('effective_trigger_step') is not None:
                effective_trigger_step = int(card.get('effective_trigger_step') or trigger_step)
            elif target_phase == 'cart_check' and card.get('recommended_card') == 'cart_reconciliation_card':
                effective_trigger_step = 1
            else:
                effective_trigger_step = trigger_step
            if target_phase != 'any' and phase != target_phase:
                return messages
            if step < effective_trigger_step:
                if not state.get('pfl_activation_wait_logged'):
                    state['pfl_activation_wait_logged'] = True
                    state['events'].append({
                        'type': 'pfl_online_intervention_skipped',
                        'phase': phase,
                        'step': step,
                        'reason': 'before_effective_trigger_step',
                        'recommended_card': card.get('recommended_card'),
                        'pfl_family': card.get('pfl_family'),
                        'trigger_step': trigger_step,
                        'effective_trigger_step': effective_trigger_step,
                    })
                return messages
            if self._persist_ablation_mode() == 'no_recoverability_gate':
                handoff_allowed = True
                handoff_state = {
                    'ablation': 'no_recoverability_gate',
                    'phase': phase,
                    'step': step,
                    'gate_bypassed': True,
                }
                handoff_skip_reason = None
            else:
                handoff_allowed, handoff_state, handoff_skip_reason = self._persist_pfl_handoff_gate(
                    state,
                    messages,
                    card,
                    phase,
                    step,
                )
            if not handoff_allowed:
                handoff_signature = (
                    handoff_skip_reason,
                    handoff_state.get('phase'),
                    handoff_state.get('solved_requirement_count'),
                    handoff_state.get('missing_requirement_count'),
                    handoff_state.get('last_lookup_tool'),
                    handoff_state.get('last_lookup_signature_count'),
                    handoff_state.get('recent_requirement_satisfaction_churn'),
                    handoff_state.get('requirement_satisfaction_stale_run'),
                )
                if card.get('trigger_mode') in {
                    'phase_handoff_unresolved_constraints',
                    'online_evidence_stagnation',
                    'online_ready_to_detail_stagnation',
                    'online_partial_coverage_stagnation',
                    'online_requirement_closure',
                } and state.get('pfl_handoff_gate_signature') != handoff_signature:
                    state['pfl_handoff_gate_logged'] = True
                    state['pfl_handoff_gate_signature'] = handoff_signature
                    state['events'].append({
                        'type': 'pfl_recoverability_gate_skipped',
                        'phase': phase,
                        'step': step,
                        'reason': handoff_skip_reason,
                        'recommended_card': card.get('recommended_card'),
                        'pfl_family': card.get('pfl_family'),
                        'gate_state': handoff_state,
                    })
                return messages
            persisted = list(messages)
            persisted.append({
                'role': 'system',
                'content': self._persist_pfl_card_message(state, card),
            })
            state['pfl_card_prompted'] = True
            state['events'].append({
                'type': 'pfl_online_intervention',
                'phase': phase,
                'step': step,
                'recommended_card': card.get('recommended_card'),
                'pfl_family': card.get('pfl_family'),
                'decision': 'one_time_typed_recovery_card_without_tool_modification',
                'target_phase': target_phase,
                'trigger_step': trigger_step,
                'effective_trigger_step': effective_trigger_step,
                'trigger_mode': card.get('trigger_mode'),
                'handoff_gate_state': handoff_state,
                'ablation': self._persist_ablation_mode() or None,
            })
            return persisted
        if state.get('enabled') and self._persist_v5_online_enabled():
            ledger = state.get('v5_requirement_ledger')
            if not ledger:
                return messages
            pending = state.get('v5_pending_ace')
            if not pending:
                return messages
            persisted = list(messages)
            persisted.append({
                'role': 'system',
                'content': (
                    '[PERSIST-v5 state] Use this parser ledger as a compact checklist of requested products and constraints. '
                    'Infer progress from the real tool history and cart. Do not invent product IDs and only use IDs returned by tools. '
                    'Do not add searches merely because the ledger is present; finish normally when available evidence and cart work are complete. '
                    'Ledger JSON: ' + json.dumps({'requirements': ledger}, ensure_ascii=False, separators=(',', ':'))
                ),
            })
            state['v5_ledger_prompt_count'] = int(state.get('v5_ledger_prompt_count', 0)) + 1
            state['events'].append({
                'type': 'v5_state_prompt',
                'phase': phase,
                'step': step,
                'reason': 'strict_no_effect_recovery',
                'requirement_count': len(ledger),
            })
            if pending:
                if self._persist_v5_ace_once_enabled() and int(state.get('v5_ace_prompt_count', 0)) >= 1:
                    state['events'].append({
                        'type': 'v5_online_intervention_skipped',
                        'phase': phase,
                        'step': step,
                        'reason': 'ace_once_budget_exhausted',
                        **pending,
                    })
                    state['v5_pending_ace'] = None
                    return persisted
                persisted.append({
                    'role': 'system',
                    'content': (
                        '[PERSIST-v5 ACE] The exact lookup below produced no functional state/evidence change twice. '
                        'Do not repeat the same tool and arguments on this turn. Continue only genuinely unresolved requirements, '
                        f"but switch to tool family `{pending['suggested_tool_family']}`. Do not finalize or stop merely "
                        'because this lookup stalled; if all requirements are already satisfied, complete the normal cart/final response. '
                        'Trigger JSON: ' + json.dumps(pending, ensure_ascii=False, separators=(',', ':'))
                    ),
                })
                state['events'].append({
                    'type': 'v5_online_intervention',
                    'phase': phase,
                    'step': step,
                    **pending,
                    'decision': 'prompt_next_turn_family_switch_without_dropping_current_call',
                })
                state['v5_ace_prompt_count'] = int(state.get('v5_ace_prompt_count', 0)) + 1
                state['v5_pending_ace'] = None
            return persisted
        if not state.get('enabled') or not self._persist_proposed_v3_enabled():
            return messages
        limit = int(os.getenv('SHOPPING_PERSIST_COMPACTION_CHAR_BUDGET', '70000'))
        before_chars = self._persist_estimate_history_chars(messages)
        if before_chars < limit:
            return messages
        summary = self._persist_build_evidence_summary(messages)
        compacted: List[Dict[str, Any]] = []
        system_message = next((message for message in messages if message.get('role') == 'system'), None)
        user_messages = [message for message in messages if message.get('role') == 'user']
        summary_card = (
            '[PERSIST-ACE] shopping_evidence_compaction_card. The JSON below is a deterministic summary '
            'of visible tool evidence. Use only listed product/coupon IDs; do not invent IDs. If supported '
            'candidates are missing from the cart, prioritize cart completion before further retrieval. '
            'Preserve current cart items that remain supported by visible evidence; only delete an item when '
            'the evidence clearly shows that it violates a stated requirement. Do not repeat a recent tool call '
            'unless its arguments or the visible evidence have materially changed.\n'
            + json.dumps(summary, ensure_ascii=False, separators=(',', ':'))
        )
        compacted.append({
            'role': 'system',
            'content': ((system_message or {}).get('content') or '') + '\n\n' + summary_card,
        })
        if user_messages:
            compacted.append(user_messages[0])
        if phase == 'cart_check' and len(user_messages) > 1:
            compacted.append(user_messages[-1])
        after_chars = self._persist_estimate_history_chars(compacted)
        signature = f'{phase}:{step}:{before_chars}:{after_chars}'
        state['context_compaction_counts'][phase] = int(state['context_compaction_counts'].get(phase, 0)) + 1
        if signature not in state['context_compaction_signatures']:
            state['context_compaction_signatures'].add(signature)
            state['events'].append({
                'type': 'intervention',
                'phase': phase,
                'step': step,
                'reason': 'shopping_evidence_compaction_card',
                'before_chars': before_chars,
                'after_chars': after_chars,
                'candidate_count': len(summary['candidate_product_ids']),
                'cart_item_count': len(summary['latest_cart'].get('items', [])),
            })
        return compacted

    def _persist_should_reprompt_cart_completion(self, state: Dict[str, Any], content: str) -> bool:
        if not state.get('enabled') or not self._persist_proposed_v3_enabled():
            return False
        if self._persist_conservative_v4_enabled():
            return False
        if int(state.get('cart_completion_reprompts', 0)) >= 1:
            return False
        normalized = str(content or '').lower()
        completed_markers = ('all requirements are met', 'fully meets all requirements', 'task is already complete')
        if any(marker in normalized for marker in completed_markers):
            return False
        repair_markers = (
            'missing from the cart',
            'cart is missing',
            'required item is missing',
        )
        return any(marker in normalized for marker in repair_markers)

    def _persist_estimate_history_chars(self, messages: List[Dict[str, Any]]) -> int:
        try:
            return len(json.dumps(messages, ensure_ascii=False))
        except Exception:
            return sum(len(str(message)) for message in messages)

    def _persist_context_budget_exceeded(self, messages: List[Dict[str, Any]]) -> bool:
        if not self._persist_proposed_v2_enabled():
            return False
        limit = int(os.getenv('SHOPPING_PERSIST_CONTEXT_CHAR_BUDGET', '90000'))
        return self._persist_estimate_history_chars(messages) >= limit

    def _persist_context_budget_message(self, phase: str) -> str:
        return (
            '[PERSIST-ACE] shopping_context_budget_finalization card: visible tool evidence is already large enough '
            'that another model call is likely to overflow the service context. Do not request more tools. '
            'Use the current cart and visible product/coupon evidence as the final state; if requirements remain unsupported, '
            'stop with the least harmful partial cart rather than repeating retrieval.'
        )

    def _persist_apply_current_turn_fanout_gate(self, state: Dict[str, Any], calls: List[Dict[str, Any]], phase: str, step: int) -> List[Dict[str, Any]]:
        if not state.get('enabled') or not self._persist_proposed_v2_enabled() or not calls:
            return calls
        default_max_calls = '12' if self._persist_proposed_v3_enabled() else '8'
        max_calls = int(os.getenv('SHOPPING_PERSIST_MAX_TOOL_CALLS_PER_TURN', default_max_calls))
        if len(calls) <= max_calls:
            return calls
        if self._persist_conservative_v4_enabled() and int(state.get('context_compaction_counts', {}).get(phase, 0)) == 0:
            return calls
        if self._persist_proposed_v3_enabled():
            families = {
                'action': [index for index, call in enumerate(calls) if call.get('name') in self._PERSIST_ACTION_TOOLS],
                'detail': [index for index, call in enumerate(calls) if call.get('name') in {'get_product_details', 'calculate_transport_time', 'get_cart_info'}],
                'retrieval': [index for index, call in enumerate(calls) if call.get('name') in self._PERSIST_LOOKUP_TOOLS and call.get('name') not in {'get_product_details', 'calculate_transport_time', 'get_cart_info'}],
            }
            assigned = set().union(*[set(indexes) for indexes in families.values()])
            families['other'] = [index for index in range(len(calls)) if index not in assigned]
            quotas = {'action': 6, 'detail': 3, 'retrieval': 4, 'other': 2}

            def spread(indexes: List[int], count: int) -> List[int]:
                if count <= 0 or not indexes:
                    return []
                if len(indexes) <= count:
                    return indexes
                if count <= 1:
                    return [indexes[0]]
                positions = {round(offset * (len(indexes) - 1) / (count - 1)) for offset in range(count)}
                return [indexes[position] for position in sorted(positions)]

            selected: set[int] = set()
            for family in ('action', 'detail', 'retrieval', 'other'):
                selected.update(spread(families[family], min(quotas[family], max_calls - len(selected))))
                if len(selected) >= max_calls:
                    break
            if len(selected) < max_calls:
                remaining = [index for index in range(len(calls)) if index not in selected]
                selected.update(spread(remaining, max_calls - len(selected)))
            kept_indexes = sorted(selected)[:max_calls]
            kept = [calls[index] for index in kept_indexes]
            dropped = [call for index, call in enumerate(calls) if index not in selected]
        else:
            kept = calls[:max_calls]
            dropped = calls[max_calls:]
        state['events'].append({
            'type': 'fanout_truncated',
            'phase': phase,
            'step': step,
            'kept': len(kept),
            'dropped': len(dropped),
            'reason': 'shopping_current_turn_tool_fanout_budget',
            'kept_tools': [call.get('name') for call in kept],
            'dropped_tools': [call.get('name') for call in dropped],
        })
        return kept

    def _persist_prepare_tool_calls(self, state: Dict[str, Any], calls: List[Dict[str, Any]], phase: str, step: int) -> tuple[List[Dict[str, Any]], Optional[str]]:
        if not state.get('enabled') or not calls:
            return calls, None
        if (state.get('action_registry') or {}).get('cards'):
            return self._persist_prepare_action_registry_calls(state, calls, phase, step), None
        if self._persist_pfl_plan_enabled():
            filtered, message = self._persist_prepare_pfl_state_guard_calls(state, calls, phase, step)
            if filtered is not None or message is not None:
                return filtered or [], message
            filtered, message = self._persist_prepare_pfl_repeat_safe_calls(state, calls, phase, step)
            if filtered is not None or message is not None:
                return filtered or [], message
            return calls, None
        if self._persist_v5_shadow_enabled():
            for call in calls:
                name = call.get('name')
                key = self._persist_call_key(call)
                no_effect_count = int(state['no_effect_lookup_counts'].get(key, 0))
                failed_count = int(state['failed_lookup_counts'].get(key, 0))
                would_trigger = bool(
                    name in self._PERSIST_LOOKUP_TOOLS
                    and (no_effect_count >= 2 or failed_count >= 2)
                )
                trigger_family = None
                suggested_family = None
                if would_trigger:
                    trigger_family = (
                        'strict_repeated_failure' if failed_count >= 2 else 'strict_exact_no_effect'
                    )
                    if name == 'search_products':
                        suggested_family = 'detail' if state.get('visible_product_ids') else 'search_change_query'
                    elif name in {'get_product_details', 'calculate_transport_time'}:
                        suggested_family = 'search_change_query'
                    else:
                        suggested_family = 'detail' if state.get('visible_product_ids') else 'search_change_query'
                event = {
                    'type': 'v5_shadow_decision',
                    'phase': phase,
                    'step': step,
                    'tool_call_id': call.get('id'),
                    'would_trigger': would_trigger,
                    'trigger_family': trigger_family,
                    'suggested_tool_family': suggested_family,
                    'actual_next_action': {
                        'tool': name,
                        'arguments': self._persist_normalize_arguments(call.get('arguments', '')),
                    },
                    'strict_no_effect_count': no_effect_count,
                    'strict_failure_count': failed_count,
                    'decision': 'observe_only_execute_original_call',
                }
                state['events'].append(event)
                state['v5_shadow_pending'][call.get('id')] = event
            return calls, None
        if self._persist_v5_online_enabled():
            return calls, None
        if self._persist_proposed_v3_enabled():
            return self._persist_prepare_tool_calls_v3(state, calls, phase, step)
        for call in calls:
            name = call.get('name')
            key = self._persist_call_key(call)
            if name in self._PERSIST_ACTION_TOOLS and key in state['failed_action_keys']:
                if self._persist_proposed_v3_enabled() and int(state.get('evidence_version', 0)) > int(state['failed_action_versions'].get(key, 0)):
                    state['events'].append({'type': 'retry_protected', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': 'productive_action_retry_after_evidence_delta'})
                    continue
                reason = 'stop_repeated_failed_domain_action'
                state['events'].append({'type': 'intervention', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': reason})
                return [], self._persist_intervention_message(reason, name)
            if name in self._PERSIST_LOOKUP_TOOLS:
                failed_count = int(state['failed_lookup_counts'].get(key, 0))
                if failed_count >= 2:
                    if self._persist_proposed_v3_enabled() and int(state.get('evidence_version', 0)) > int(state['failed_evidence_versions'].get(key, 0)):
                        state['events'].append({'type': 'retry_protected', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': 'productive_lookup_retry_after_evidence_delta', 'failed_count': failed_count})
                        continue
                    reason = 'stop_repeated_failed_domain_lookup'
                    state['events'].append({'type': 'intervention', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': reason, 'failed_count': failed_count})
                    return [], self._persist_intervention_message(reason, name)
                if failed_count == 1:
                    state['events'].append({'type': 'retry_protected', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': 'first_failed_lookup_repeat_allowed'})
        return calls, None

    def _persist_prepare_action_registry_calls(
        self,
        state: Dict[str, Any],
        calls: List[Dict[str, Any]],
        phase: str,
        step: int,
    ) -> List[Dict[str, Any]]:
        triggered_cards = state['action_registry_triggered_cards']
        for card in state['action_registry']['cards']:
            card_id = str(card['card_id'])
            if card_id in triggered_cards:
                continue
            for call_index, call in enumerate(calls):
                try:
                    arguments = json.loads(call.get('arguments') or '{}')
                except Exception:
                    continue
                replacement_tool = None
                replacement_arguments = None
                trigger_evidence = None

                if (
                    card_id in {
                        'shopping_rating_distribution_key_alias_online_v2',
                        'shopping_f3_train_rating_key_rebinding_v1',
                        'shopping_f3_rerun_tradeoff_rating_key_rebinding_v1',
                    }
                    and call.get('name') == 'filter_by_range'
                    and arguments.get('product_ids')
                ):
                    condition_key = str(arguments.get('condition_key') or '')
                    replacement_key = self._PERSIST_RATING_KEY_ALIASES.get(condition_key)
                    if replacement_key:
                        replacement_tool = 'filter_by_range'
                        replacement_arguments = dict(arguments)
                        replacement_arguments['condition_key'] = replacement_key
                        trigger_evidence = 'official_tool_schema_unique_alias'

                elif (
                    card_id == 'shopping_coupon_name_format_alias_from_prior_error_v1'
                    and call.get('name') == 'add_coupon_to_cart'
                ):
                    coupon_name = str(arguments.get('coupon_name') or '')
                    exact_coupon = state['coupon_aliases_from_prior_errors'].get(
                        self._persist_normalize_text(coupon_name)
                    )
                    if exact_coupon and exact_coupon != coupon_name:
                        replacement_tool = 'add_coupon_to_cart'
                        replacement_arguments = dict(arguments)
                        replacement_arguments['coupon_name'] = exact_coupon
                        trigger_evidence = 'prior_tool_error_unique_coupon_spelling'

                elif (
                    card_id == 'shopping_repeated_failed_coupon_action_to_cart_check_v1'
                    and call.get('name') == 'add_coupon_to_cart'
                    and self._persist_call_key(call) in state['failed_action_keys']
                ):
                    replacement_tool = 'get_cart_info'
                    replacement_arguments = {}
                    trigger_evidence = 'prior_exact_coupon_action_failure'

                elif (
                    card_id in {
                        'shopping_preserve_nonempty_cart_delete_to_cart_check_v1',
                        'shopping_f3_train_second_delete_checkpoint_v1',
                        'shopping_f3_rerun_tradeoff_second_delete_checkpoint_v1',
                    }
                    and call.get('name') == 'delete_product_from_cart'
                    and state.get('nonempty_cart_seen')
                    and int(state.get('delete_calls_after_nonempty_cart', 0)) >= 1
                ):
                    replacement_tool = 'get_cart_info'
                    replacement_arguments = {}
                    trigger_evidence = 'prior_nonempty_cart_and_prior_delete_action'

                elif (
                    card_id == 'shopping_f3_train_transport_name_prerequisite_search_v1'
                    and call.get('name') == 'calculate_transport_time'
                    and self._persist_product_name_like(arguments.get('product_id'))
                ):
                    replacement_tool = 'search_products'
                    replacement_arguments = {
                        'query': str(arguments['product_id']).strip(),
                        'limit': 5,
                    }
                    trigger_evidence = 'official_transport_schema_requires_product_id'

                if replacement_tool is None or replacement_arguments is None:
                    continue

                replacement_calls = [dict(item) for item in calls]
                replacement_calls[call_index] = {
                    'id': call.get('id'),
                    'name': replacement_tool,
                    'arguments': json.dumps(replacement_arguments, ensure_ascii=False),
                }
                original_action = {
                    'tool': call.get('name'),
                    'arguments': arguments,
                }
                replacement_action = {
                    'tool': replacement_tool,
                    'arguments': replacement_arguments,
                }
                pre_action_state_hash = hashlib.sha256(
                    json.dumps({
                        'sample_id': state.get('sample_id'),
                        'phase': phase,
                        'step': step,
                        'evidence_version': state.get('evidence_version', 0),
                        'calls': calls,
                    }, ensure_ascii=False, sort_keys=True).encode('utf-8')
                ).hexdigest()
                state['events'].append({
                    'type': 'action_registry_replacement',
                    'registry_id': state['action_registry']['registry_id'],
                    'card_id': card_id,
                    'phase': phase,
                    'step': step,
                    'call_index': call_index,
                    'pre_action_state_hash': pre_action_state_hash,
                    'trigger_evidence': trigger_evidence,
                    'original_action': original_action,
                    'replacement_action': replacement_action,
                    'decision': 'execute_single_materialized_replacement',
                })
                triggered_cards.add(card_id)
                return replacement_calls
        return calls

    def _persist_prepare_pfl_repeat_safe_calls(self, state: Dict[str, Any], calls: List[Dict[str, Any]], phase: str, step: int) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        card = state.get('pfl_card')
        if not isinstance(card, dict):
            return None, None
        repeat_safe_card = (
            card.get('card_variant') == 'constraint_backtracking_repeat_safe'
            or (
                self._persist_pfl_cb_repeat_safe_enabled()
                and card.get('recommended_card') == 'constraint_backtracking_card'
            )
        )
        if not repeat_safe_card:
            return None, None
        prepared: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        last_key = state.get('last_executed_call_key')
        for call in calls:
            name = call.get('name')
            key = self._persist_call_key(call)
            if name in self._PERSIST_LOOKUP_TOOLS and key == last_key:
                state['events'].append({
                    'type': 'pfl_repeat_safe_intervention',
                    'phase': phase,
                    'step': step,
                    'tool': name,
                    'key': key,
                    'reason': 'drop_adjacent_identical_lookup_call',
                    'decision': 'drop_only_repeated_lookup_keep_other_calls',
                    'card_variant': card.get('card_variant'),
                })
                blocked.append(call)
                continue
            prepared.append(call)
        if prepared:
            return prepared, None
        if blocked:
            return [], (
                '[PERSIST-ACE-Plan repeat-safe] The immediately previous lookup call used identical arguments. '
                'Do not repeat that same call. Switch to a different unresolved constraint, inspect details for visible candidates, '
                'read the cart, or finish if no evidence-backed action remains.'
            )
        return None, None

    def _persist_prepare_pfl_state_guard_calls(self, state: Dict[str, Any], calls: List[Dict[str, Any]], phase: str, step: int) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        if not self._persist_pfl_state_guard_enabled():
            return None, None
        card = state.get('pfl_card')
        if not isinstance(card, dict):
            return None, None
        if card.get('recommended_card') not in {
            'evidence_refresh_card',
            'coverage_completion_card',
            'requirement_closure_card',
        }:
            return None, None
        if not state.get('pfl_card_prompted'):
            if not self._persist_pfl_state_guard_precard_enabled():
                return None, None
            min_step = int(os.getenv('SHOPPING_PERSIST_PFL_STATE_GUARD_PRECARD_MIN_STEP', '1'))
            if step < min_step:
                return None, None
        prepared: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        allowed_range_keys: set[str] | None = None
        budget_window: tuple[float | None, float | None] = (None, None)
        if (
            card.get('recommended_card') == 'requirement_closure_card'
            and state.get('v5_active_closure_subtype')
        ):
            unresolved_ids = {
                str(requirement.get('requirement_id'))
                for requirement in (state.get('v5_requirement_state') or {}).get('requirements', [])
                if not requirement.get('solved') and requirement.get('requirement_id')
            }
            unresolved_definitions = [
                requirement
                for requirement in state.get('v5_requirement_ledger', [])
                if str(requirement.get('requirement_id')) in unresolved_ids
            ]
            allowed_range_keys = set()
            for requirement in unresolved_definitions:
                for _, path, _, _ in numeric_constraints(text_for_requirement(requirement)):
                    if isinstance(path, tuple):
                        allowed_range_keys.add('.'.join(path))
            if len(unresolved_definitions) == 1:
                budget_window = remaining_unit_price_window(
                    state.get('user_query', ''),
                    state.get('latest_cart_total'),
                    int(unresolved_definitions[0].get('quantity', 1)),
                )
                if budget_window != (None, None):
                    allowed_range_keys.add('price')
        ranked_candidate_ids = (
            rank_supported_products(
                unresolved_definitions[0],
                state.get('v5_requirement_products', {}),
                state.get('v5_requirement_transport_times', {}),
                *budget_window,
            )
            if allowed_range_keys is not None and len(unresolved_definitions) == 1
            else []
        )
        closure_lookup_calls = [
            call for call in calls if call.get('name') in self._PERSIST_LOOKUP_TOOLS
        ]
        new_planning_recovery = state.get('v5_trigger_closure_subtype') in {
            'REPEATED_CART_CHECK_MISSING_EVIDENCE',
            'COUPON_OSCILLATION_MISSING_PREREQUISITE',
        }
        if (
            state.get('v5_active_closure_subtype') == 'SERIALIZED_REQUIREMENT_CLOSURE'
            and int(state.get('serialized_closure_lookup_count', 0)) == 0
            and closure_lookup_calls
        ):
            first_lookup = closure_lookup_calls[0]
            raw_arguments = first_lookup.get('arguments', '')
            if isinstance(raw_arguments, dict):
                first_arguments = raw_arguments
            else:
                try:
                    first_arguments = json.loads(raw_arguments or '{}')
                except (TypeError, ValueError):
                    first_arguments = {}
            first_key = str(first_arguments.get('condition_key') or '')
            invalid_first_lookup = (
                first_lookup.get('name') != 'filter_by_range'
                or bool(first_arguments.get('product_ids'))
                or first_key not in (allowed_range_keys or set())
            )
            if invalid_first_lookup:
                state['events'].append({
                    'type': 'pfl_state_guard_intervention',
                    'phase': phase,
                    'step': step,
                    'tool': first_lookup.get('name'),
                    'reason': 'require_global_range_reset_before_serialized_closure',
                    'key': self._persist_call_key(first_lookup),
                    'allowed_range_keys': sorted(allowed_range_keys or []),
                    'decision': 'block_stale_subset_lookup_without_executing_tool',
                })
                return [], (
                    '[PERSIST-ACE-Plan serialized closure] The first recovery lookup must reset stale candidate '
                    'state: call filter_by_range once without product_ids, using one allowed_range_key from the '
                    'sole unresolved requirement. Wait for its real returned IDs before the next filter.'
                )
        if (
            new_planning_recovery
            and len(state.get('last_retrieval_product_ids') or set()) > 3
            and any(call.get('name') == 'get_product_details' for call in closure_lookup_calls)
        ):
            state['events'].append({
                'type': 'pfl_state_guard_intervention',
                'phase': phase,
                'step': step,
                'tool': 'get_product_details',
                'reason': 'require_complete_constraint_narrowing_before_details',
                'candidate_count': len(state.get('last_retrieval_product_ids') or set()),
                'decision': 'block_premature_details_without_selecting_or_injecting_product_id',
            })
            return [], (
                '[PERSIST-ACE-Planning prerequisite guard] Too many candidates remain for a grounded choice. '
                'Re-read the original user request and apply one still-unused identity, brand, color, size, or '
                'numeric constraint to the exact IDs returned by the previous lookup. Wait for that result; '
                'inspect details only after at most three candidates remain.'
            )
        if allowed_range_keys is not None and len(closure_lookup_calls) > 1:
            kept_lookup = closure_lookup_calls[0]
            dropped_lookup_calls = closure_lookup_calls[1:]
            state['events'].append({
                'type': 'pfl_state_guard_intervention',
                'phase': phase,
                'step': step,
                'tool': kept_lookup.get('name'),
                'reason': 'serialize_requirement_closure_lookup_chain',
                'kept_key': self._persist_call_key(kept_lookup),
                'dropped_keys': [self._persist_call_key(call) for call in dropped_lookup_calls],
                'decision': 'execute_one_lookup_then_condition_next_lookup_on_real_result',
            })
            calls = [kept_lookup]
        for call in calls:
            name = call.get('name')
            key = self._persist_call_key(call)
            if (
                state.get('pfl_coupon_completion_done')
                and name in {'add_coupon_to_cart', 'delete_coupon_from_cart'}
            ):
                state['events'].append({
                    'type': 'pfl_state_guard_intervention',
                    'phase': phase,
                    'step': step,
                    'tool': name,
                    'key': key,
                    'reason': 'drop_coupon_mutation_after_successful_requirement_closure',
                    'decision': 'preserve_first_successful_post_closure_coupon_state',
                })
                blocked.append(call)
                continue
            if name == 'add_product_to_cart' and (ranked_candidate_ids or new_planning_recovery):
                raw_arguments = call.get('arguments', '')
                if isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    try:
                        arguments = json.loads(raw_arguments or '{}')
                    except (TypeError, ValueError):
                        arguments = {}
                proposed_product_id = str(arguments.get('product_id') or '')
                unsupported_new_recovery_write = (
                    new_planning_recovery
                    and proposed_product_id not in ranked_candidate_ids
                )
                non_boundary_best_write = (
                    not new_planning_recovery
                    and ranked_candidate_ids
                    and proposed_product_id != ranked_candidate_ids[0]
                )
                if unsupported_new_recovery_write or non_boundary_best_write:
                    reason = (
                        'drop_unverified_recovery_cart_write'
                        if new_planning_recovery
                        else 'drop_non_boundary_best_cart_write'
                    )
                    state['events'].append({
                        'type': 'pfl_state_guard_intervention',
                        'phase': phase,
                        'step': step,
                        'tool': name,
                        'key': key,
                        'reason': reason,
                        'proposed_product_id': proposed_product_id,
                        'boundary_best_candidate_id': ranked_candidate_ids[0] if ranked_candidate_ids else None,
                        'ranked_candidate_ids': ranked_candidate_ids[:3],
                        'decision': 'block_wrong_write_without_injecting_cart_action',
                        'recommended_card': card.get('recommended_card'),
                    })
                    blocked.append(call)
                    continue
            if name not in self._PERSIST_LOOKUP_TOOLS:
                prepared.append(call)
                continue
            no_effect_count = int(state['no_effect_lookup_counts'].get(key, 0))
            failed_count = int(state['failed_lookup_counts'].get(key, 0))
            authoritative_failure = key in state.get('authoritative_failed_lookup_keys', set())
            evidence_version = int(state.get('evidence_version', 0))
            no_effect_version = int(state['no_effect_evidence_versions'].get(key, -1))
            failed_version = int(state['failed_evidence_versions'].get(key, -1))
            outside_unresolved_requirement = False
            if allowed_range_keys is not None and name == 'filter_by_range':
                raw_arguments = call.get('arguments', '')
                if isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    try:
                        arguments = json.loads(raw_arguments or '{}')
                    except (TypeError, ValueError):
                        arguments = {}
                condition_key = str(arguments.get('condition_key') or '')
                outside_unresolved_requirement = condition_key not in allowed_range_keys
            should_block = (
                outside_unresolved_requirement
                or authoritative_failure
                or (no_effect_count >= 1 and evidence_version <= no_effect_version)
                or (failed_count >= 1 and evidence_version <= failed_version)
            )
            if should_block:
                state['events'].append({
                    'type': 'pfl_state_guard_intervention',
                    'phase': phase,
                    'step': step,
                    'tool': name,
                    'key': key,
                    'reason': (
                        'drop_lookup_outside_unresolved_requirement'
                        if outside_unresolved_requirement
                        else 'drop_authoritatively_invalid_lookup_after_er_card'
                        if authoritative_failure
                        else 'drop_lookup_without_new_state_progress_after_er_card'
                    ),
                    'no_effect_count': no_effect_count,
                    'failed_count': failed_count,
                    'authoritative_failure': authoritative_failure,
                    'evidence_version': evidence_version,
                    'decision': 'drop_only_stale_lookup_keep_other_calls',
                    'recommended_card': card.get('recommended_card'),
                    'allowed_range_keys': sorted(allowed_range_keys or []),
                })
                blocked.append(call)
                continue
            prepared.append(call)
        if prepared:
            return prepared, None
        if blocked:
            boundary_events = [
                event
                for event in state.get('events', [])[-len(blocked):]
                if event.get('reason') == 'drop_non_boundary_best_cart_write'
            ]
            if boundary_events:
                best_id = boundary_events[-1]['boundary_best_candidate_id']
                return [], (
                    '[PERSIST-ACE-Plan requirement guard] The proposed cart write was blocked because the product '
                    'was not boundary-best under the unresolved numeric constraints. Re-evaluate the grounded '
                    f'candidates; the online-visible boundary-best candidate is {best_id}. You retain the decision '
                    'whether to add it. Do not add the blocked product.'
                )
            if any(
                event.get('reason') == 'drop_unverified_recovery_cart_write'
                for event in state.get('events', [])[-len(blocked):]
            ):
                return [], (
                    '[PERSIST-ACE-Planning write guard] The proposed product is not yet fully supported by visible '
                    'evidence for every unresolved requirement constraint. No product ID was selected or injected. '
                    'Continue read-only narrowing/details, or finish without changing the cart.'
                )
            if any(
                event.get('reason') == 'drop_coupon_mutation_after_successful_requirement_closure'
                for event in state.get('events', [])[-len(blocked):]
            ):
                return [], (
                    '[PERSIST-ACE-Plan repeat-safe] Product requirements are closed and the first new coupon '
                    'was added successfully. Preserve this coupon state; do not add or delete more coupons. '
                    'Inspect the cart once or finish.'
                )
            return [], (
                '[PERSIST-ACE-Plan state guard] The requested lookup has already failed or produced no new evidence since the recovery card. '
                'Do not repeat it. Change to a different unresolved constraint, inspect details for visible candidates, read the cart, '
                'or finish if no evidence-backed action remains.'
            )
        return None, None

    def _persist_prepare_tool_calls_v3(self, state: Dict[str, Any], calls: List[Dict[str, Any]], phase: str, step: int) -> tuple[List[Dict[str, Any]], Optional[str]]:
        compaction_count = int(state.get('context_compaction_counts', {}).get(phase, 0))
        if self._persist_conservative_v4_enabled() and compaction_count == 0:
            return calls, None
        if compaction_count >= 4:
            action_calls = [call for call in calls if call.get('name') in self._PERSIST_ACTION_TOOLS]
            if action_calls:
                state['events'].append({
                    'type': 'intervention',
                    'phase': phase,
                    'step': step,
                    'reason': 'shopping_context_budget_router_action_only',
                    'compaction_count': compaction_count,
                    'kept_tools': [call.get('name') for call in action_calls],
                    'dropped_tools': [call.get('name') for call in calls if call not in action_calls],
                })
                calls = action_calls
            else:
                reason = 'shopping_context_budget_router_to_cart' if phase == 'search' else 'shopping_context_budget_router_safe_stop'
                state['events'].append({
                    'type': 'intervention',
                    'phase': phase,
                    'step': step,
                    'reason': reason,
                    'compaction_count': compaction_count,
                    'dropped_tools': [call.get('name') for call in calls],
                })
                return [], self._persist_context_router_message(phase)
        prepared: List[Dict[str, Any]] = []
        blocked_reasons: List[tuple[str, str]] = []
        for call in calls:
            name = call.get('name')
            key = self._persist_call_key(call)
            if name in self._PERSIST_ACTION_TOOLS and key in state['failed_action_keys']:
                if int(state.get('evidence_version', 0)) > int(state['failed_action_versions'].get(key, 0)):
                    state['events'].append({'type': 'retry_protected', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': 'productive_action_retry_after_evidence_delta'})
                    prepared.append(call)
                    continue
                reason = 'stop_repeated_failed_domain_action'
                state['events'].append({'type': 'intervention', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': reason, 'decision': 'drop_only_failed_action_keep_other_calls'})
                blocked_reasons.append((reason, name))
                continue
            if name in self._PERSIST_LOOKUP_TOOLS:
                no_effect_count = int(state['no_effect_lookup_counts'].get(key, 0))
                if no_effect_count >= 1:
                    if int(state.get('evidence_version', 0)) > int(state['no_effect_evidence_versions'].get(key, 0)):
                        state['events'].append({'type': 'retry_protected', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': 'productive_lookup_retry_after_evidence_delta', 'no_effect_count': no_effect_count})
                    else:
                        reason = 'stop_repeated_no_effect_lookup'
                        state['events'].append({'type': 'intervention', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': reason, 'no_effect_count': no_effect_count, 'decision': 'drop_only_no_effect_lookup_keep_other_calls'})
                        blocked_reasons.append((reason, name))
                        continue
                failed_count = int(state['failed_lookup_counts'].get(key, 0))
                if failed_count == 1:
                    state['events'].append({'type': 'retry_protected', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': 'first_failed_lookup_repeat_allowed'})
                if failed_count >= 2:
                    if int(state.get('evidence_version', 0)) > int(state['failed_evidence_versions'].get(key, 0)):
                        state['events'].append({'type': 'retry_protected', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': 'productive_lookup_retry_after_evidence_delta', 'failed_count': failed_count})
                        prepared.append(call)
                        continue
                    repair = self._persist_failed_lookup_repair_call(call)
                    if repair is not None:
                        state['events'].append({'type': 'intervention', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': 'shopping_failed_filter_to_details_repair', 'failed_count': failed_count, 'repair_tool': repair.get('name')})
                        prepared.append(repair)
                        continue
                    reason = 'stop_repeated_failed_domain_lookup'
                    state['events'].append({'type': 'intervention', 'phase': phase, 'step': step, 'tool': name, 'key': key, 'reason': reason, 'failed_count': failed_count, 'decision': 'drop_only_failed_lookup_keep_other_calls'})
                    blocked_reasons.append((reason, name))
                    continue
            prepared.append(call)
        unique: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for call in prepared:
            key = self._persist_call_key(call)
            if key in seen:
                continue
            seen.add(key)
            unique.append(call)
        if unique:
            return unique, None
        if blocked_reasons:
            reason, tool_name = blocked_reasons[0]
            return [], self._persist_intervention_message(reason, tool_name)
        return [], None

    def _persist_context_router_message(self, phase: str) -> str:
        if phase == 'search':
            return (
                '[PERSIST-ACE] shopping_context_budget_router: the compacted evidence has already received three '
                'recovery decisions without a cart action. Stop further retrieval and proceed to cart completion '
                'using only supported visible product and coupon IDs.'
            )
        return (
            '[PERSIST-ACE] shopping_context_budget_router: repeated compacted cart checks produced no further cart '
            'action. Preserve the supported current cart and stop rather than continuing a no-effect loop.'
        )

    def _persist_failed_lookup_repair_call(self, call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if call.get('name') not in {'filter_by_range', 'filter_by_brand', 'filter_by_color', 'filter_by_size', 'sort_products'}:
            return None
        try:
            arguments = json.loads(call.get('arguments') or '{}')
        except Exception:
            return None
        product_ids = arguments.get('product_ids')
        if not isinstance(product_ids, list) or not product_ids:
            return None
        return {
            'id': call.get('id') or f'persist-repair-{uuid.uuid4().hex}',
            'name': 'get_product_details',
            'arguments': json.dumps({'product_ids': product_ids[:20]}, ensure_ascii=False),
        }

    def _persist_intervention_message(self, reason: str, tool_name: str) -> str:
        mode = os.getenv('SHOPPING_PERSIST_GUARD', '').strip().lower()
        if 'ace' in mode:
            if 'lookup' in reason:
                return (
                    f"[PERSIST-ACE] shopping_lookup_repair_or_safe_stop card ({reason}): "
                    f"the exact `{tool_name}` call has already failed repeatedly. Do not repeat the same arguments. "
                    "Use only the visible task requirements, previous observations, and cart state. If enough product evidence exists, "
                    "choose a supported alternative and proceed to cart checking; otherwise change one necessary search/filter parameter once. "
                    "If the blocker cannot be repaired from visible evidence, stop with a clear caveat."
                )
            return (
                f"[PERSIST-ACE] shopping_cart_action_repair_or_safe_stop card ({reason}): "
                f"the exact `{tool_name}` action has already failed. Do not repeat the same action. "
                "Inspect the visible cart/product evidence, repair the missing prerequisite once if possible, choose a safe alternative item/coupon, "
                "or stop with a clear caveat if no supported repair exists."
            )
        if 'cost' in mode:
            return (
                f"[PERSIST-CostGuard] conservative no-progress guard ({reason}): "
                f"the requested `{tool_name}` path is repeatedly consuming calls without useful new evidence. "
                "Stop spending turns on the same path; use gathered evidence, repair one necessary parameter, switch strategy once, "
                "or report the blocker with a caveat."
            )
        return (
            f"[PERSIST-Guard] strict repeated-failure guard ({reason}): the requested repeated tool call `{tool_name}` "
            "has already failed without adding useful information. Do not repeat the same call. "
            "Use a different search/filter/cart strategy if possible, or provide the final answer based on the current cart state."
        )

    def _persist_save_audit(self, state: Dict[str, Any]) -> None:
        audit_file = state.get('audit_file')
        if not state.get('enabled') or not audit_file:
            return
        serializable = {
            'guard': os.getenv('SHOPPING_PERSIST_GUARD', ''),
            'sample_id': self.sample_id,
            'model': self.model,
            'action_registry': state.get('action_registry', {}),
            'events': state.get('events', []),
        }
        try:
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            thread_info = threading.current_thread().name
            print(f"  ⚠️  [{thread_info}] Failed to save PERSIST audit: {e}")

    def _save_messages(self, messages: List[Any], filepath: Path, step: int, description: str):
        """Save messages to file"""
        serializable_messages = [m.model_dump() if hasattr(m, 'model_dump') else m for m in messages]
        save_data = {"step": step, "description": description, "messages": serializable_messages}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            thread_info = threading.current_thread().name
            print(f"  💾 [{thread_info}] Step {step}: {description} - Saved {len(messages)} messages")
        except Exception as e:
            thread_info = threading.current_thread().name
            print(f"  ⚠️  [{thread_info}] Failed to save messages: {e}")


def run_agent_inference(
    model: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    system_prompt: str,
    workers: int = 10,
    max_llm_calls: int = 100,
    rerun_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run agent inference (batch processing)
    
    Args:
        model: Configuration name from models_config.json
        test_data_path: Path to test data JSON file
        database_dir: Base path to database directory
        tool_schema_path: Path to tool schema JSON file
        system_prompt: System prompt for the agent
        workers: Number of parallel workers
        max_llm_calls: Maximum LLM calls per sample
        rerun_ids: Optional list of specific IDs to rerun. If None, run all samples.
    
    Returns:
        Results summary dict
    """
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Filter samples if rerun_ids is specified
    if rerun_ids is not None:
        rerun_ids_set = set(str(id) for id in rerun_ids)  # Convert to strings for comparison
        original_count = len(test_data)
        test_data = [s for s in test_data if str(s.get('id')) in rerun_ids_set]
        print(f"  🔄 Filtered {original_count} samples to {len(test_data)} samples for rerun")
        
        if len(test_data) == 0:
            print(f"  ⚠️  Warning: No samples found matching the specified IDs")
            return {
                'total': 0,
                'success': 0,
                'failed': 0,
                'elapsed_time': 0,
                'results': []
            }
    
    print(f"\n{'='*80}")
    print(f"Agent Inference")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Samples: {len(test_data)}")
    print(f"Workers: {workers}")
    print(f"{'='*80}\n")
    
    print_lock = Lock()
    results = []
    
    def process_sample(sample):
        sample_id = sample.get('id', 'unknown')
        query = sample.get('query', '')
        
        try:
            
            agent = ShoppingFnAgent(
                model=model,
                sample_id=str(sample_id),
                database_base_path=str(database_dir),
                tool_schema_path=str(tool_schema_path)
            )
            
            start_time = time.time()
            
            messages = agent.run(
                user_query=query,
                system_prompt=system_prompt,
                save_messages=True,
                sample_id=str(sample_id),
                max_llm_calls=max_llm_calls
            )
            
            elapsed = time.time() - start_time
            
            result = {
                'id': sample_id,
                'query': query,
                'model': model,
                'messages': messages,
                'elapsed_time': elapsed,
                'success': True,
            }
            
            with print_lock:
                print(f"✅ Sample {sample_id} completed in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            with print_lock:
                print(f"❌ Sample {sample_id} failed: {e}")
                import traceback
                traceback.print_exc()
            
            return {
                'id': sample_id,
                'query': query,
                'success': False,
                'error': str(e),
            }
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in test_data]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    success_count = sum(1 for r in results if r['success'])
    
    return {
        'total': len(results),
        'success': success_count,
        'failed': len(results) - success_count,
        'results': results
    }


if __name__ == '__main__':
    """Simple test"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen-plus', help='Configuration name from models_config.json')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2, 3], help='Shopping level: 1, 2, or 3')
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent
    test_output_dir = base_dir / 'results' / 'test'
    
    # Get system prompt for the specified level
    try:
        from .prompts import prompt_lib
    except ImportError:
        from prompts import prompt_lib
    
    system_prompt = getattr(prompt_lib, f'SYSTEM_PROMPT_level{args.level}', None)
    if system_prompt is None:
        raise ValueError(f"System prompt for level {args.level} not found")
    
    result = run_agent_inference(
        model=args.model,
        test_data_path=base_dir / 'data' / f'level_{args.level}_query_meta.json',
        database_dir=base_dir / 'database',
        tool_schema_path=base_dir / 'tools' / 'shopping_tool_schema.json',
        system_prompt=system_prompt,
        workers=2,
        max_llm_calls=100,
    )
    print(f"\nTest completed: {result['success']}/{result['total']} succeeded")
