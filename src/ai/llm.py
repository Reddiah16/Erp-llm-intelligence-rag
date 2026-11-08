"""One thin client for the local language model.

Everything the assistant asks a model goes through here, for three reasons.

* Availability is answered once and cached. Probing Ollama on every question
  would add a network round trip to each reply, and probing it never would mean
  a stopped server surfaces as a stack trace instead of a degraded answer with
  an honest reason.
* Model output is parsed here, tolerant of the fences and prose a small model
  wraps around the JSON it was told to produce, so no caller has to guess how
  to read it.
* Tests need one seam. Routing and answering are verified with an injected fake
  and never spend a second of GPU time or a cent of API credit.

No key exists in this module and none is needed: Ollama is a local server.
"""
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from src.ai.settings import AiSettings, default_settings

# Fences and leading prose are what a small model actually returns, however
# firmly the prompt asks for JSON only.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

SQLISH = ("select ", "insert ", "update ", "delete ", "drop ", "truncate ",
          "grant ", "alter ", "exec ", "--", ";")


class LlmUnavailable(RuntimeError):
    """The model could not be reached, or the configured model is not pulled."""


class LlmProtocolError(RuntimeError):
    """The model answered, but not as JSON. Reported, never guessed around."""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """The first JSON object in a reply, or None.

    Tries the whole string, then any fenced block, then a brace-balanced scan
    that respects quotes and escapes so a `{"a": "b}"}` inside a string does not
    end the object early. Never repairs the model's JSON: an object that cannot
    be parsed is a protocol failure the caller reports.
    """
    if not text:
        return None
    candidates: List[str] = [text.strip()]
    candidates += [block.strip() for block in _FENCE.findall(text)]
    for candidate in candidates:
        parsed = _balanced_objects(candidate)
        for fragment in parsed:
            try:
                value = json.loads(fragment)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _balanced_objects(text: str) -> List[str]:
    """Every top-level {...} span in the text, in order of appearance."""
    spans, depth, start, inside, escaped = [], 0, -1, False, False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            inside = not inside
            continue
        if inside:
            continue
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start:index + 1])
                start = -1
    return spans


def looks_like_sql(text: str) -> bool:
    """Whether a payload carries SQL the caller must not run.

    Used by the router to record a refusal. The assistant has no path that
    executes a string from the model, so this is defence in depth and a
    visible signal in the response, not the actual barrier.
    """
    lowered = f" {str(text).lower()} "
    return any(token in lowered for token in SQLISH)


class LlmClient:
    """Generate text and JSON through a local Ollama server."""

    def __init__(self, settings: Optional[AiSettings] = None,
                 client: Any = None) -> None:
        self.settings = settings or default_settings()
        self._client = client
        self._probes: Dict[str, Tuple[bool, str, float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- plumbing
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:
                raise LlmUnavailable(
                    "the 'ollama' package is not installed in this interpreter")
            self._client = ollama.Client(host=self.settings.ollama_base_url,
                                         timeout=self.settings.llm_timeout)
        return self._client

    def _remember(self, model: str, entry: Tuple[bool, str, float]) -> None:
        with self._lock:
            self._probes[model] = entry

    def cached_availability(self, model: str) -> Optional[Tuple[bool, str]]:
        """The last probe for a model, if it is still inside the TTL."""
        with self._lock:
            entry = self._probes.get(model)
        if entry is None:
            return None
        ok, reason, stamp = entry
        if time.time() - stamp > self.settings.availability_ttl:
            return None
        return ok, reason

    # ----------------------------------------------------------- availability
    def models(self) -> List[str]:
        try:
            listed = self.client.list()
        except Exception as exc:  # noqa: BLE001 - reported as unavailable
            raise LlmUnavailable(
                f"Ollama at {self.settings.ollama_base_url} is not reachable "
                f"({type(exc).__name__})")
        return [entry.get("name", "") for entry in listed.get("models", [])]

    def availability(self, model: Optional[str] = None,
                     refresh: bool = False) -> Tuple[bool, str]:
        """(usable, reason) for one model, cached for availability_ttl seconds.

        Distinguishes the two failures that look identical from a traceback:
        the server is down, and the server is up but does not have that model.
        The second one is a configuration mistake and says which models exist.
        """
        name = model or self.settings.chat_model
        if not refresh:
            cached = self.cached_availability(name)
            if cached is not None:
                return cached
        try:
            available = self.models()
        except LlmUnavailable as exc:
            entry = (False, str(exc), time.time())
            self._remember(name, entry)
            return entry[0], entry[1]
        if name in available:
            entry = (True, f"{name} is available", time.time())
        else:
            entry = (False,
                     f"model {name!r} is not pulled on this Ollama server "
                     f"(available: {', '.join(available) or 'none'})",
                     time.time())
        self._remember(name, entry)
        return entry[0], entry[1]

    # ------------------------------------------------------------- generation
    def complete(self, prompt: str, model: Optional[str] = None,
                 json_mode: bool = False,
                 max_tokens: Optional[int] = None) -> str:
        """One completion. Raises LlmUnavailable rather than returning prose
        that a caller might mistake for an answer.

        `num_predict` is always sent. Without it a small model handed a table
        can decide to reproduce the table, and the request then ends in a read
        timeout instead of an answer - a slow failure is still a failure, and
        the timeout is the only bound the caller had.
        """
        name = model or self.settings.chat_model
        ok, reason = self.availability(name)
        if not ok:
            raise LlmUnavailable(reason)
        options: Dict[str, Any] = {"temperature": 0}
        budget = self.settings.max_output_tokens if max_tokens is None else max_tokens
        if budget > 0:
            options["num_predict"] = budget
        kwargs: Dict[str, Any] = {
            "model": name,
            "prompt": prompt,
            "options": options,
        }
        if json_mode:
            kwargs["format"] = "json"
        try:
            reply = self.client.generate(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced as unavailable
            raise LlmUnavailable(
                f"generation with {name} failed ({type(exc).__name__}): "
                f"{str(exc)[:200]}")
        return (reply.get("response") or "").strip()

    def complete_json(self, prompt: str, model: Optional[str] = None,
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        text = self.complete(prompt, model=model, json_mode=True,
                             max_tokens=max_tokens)
        parsed = extract_json(text)
        if parsed is None:
            raise LlmProtocolError(
                f"{model or self.settings.route_model} did not return a JSON "
                f"object; reply began {text[:120]!r}")
        return parsed
