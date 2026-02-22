"""Provider registry: metadata for known providers and factory function.

Each ProviderSpec holds the metadata needed to auto-detect, configure,
and instantiate an LLM provider from the user's config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from grip.config.schema import GripConfig, ProviderEntry
from grip.providers.types import LLMProvider


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Metadata for a single LLM provider."""

    name: str
    display_name: str
    api_base: str
    api_key_env: str
    default_models: list[str] = field(default_factory=list)
    model_prefix: str = ""


PROVIDERS: tuple[ProviderSpec, ...] = (
    # Cloud providers (ordered per TASKS.md)
    ProviderSpec(
        name="openrouter",
        display_name="OpenRouter",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_models=[
            "anthropic/claude-opus-4.6",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-5.2",
            "openai/gpt-5.1",
            "openai/gpt-5-mini",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini",
            "openai/gpt-4o-mini",
            "openai/gpt-oss-120b",
            "openai/gpt-5-nano",
            "openai/gpt-oss-20b",
            "x-ai/grok-4.1-fast",
            "minimax/minimax-m2.5",
            "moonshotai/kimi-k2.5",
            "z-ai/glm-5",
            "z-ai/glm-4.7",
        ],
        model_prefix="openrouter/",
    ),
    ProviderSpec(
        name="anthropic",
        display_name="Anthropic",
        api_base="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        default_models=["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
        model_prefix="anthropic/",
    ),
    ProviderSpec(
        name="openai",
        display_name="OpenAI",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_models=["gpt-4o", "gpt-4o-mini", "o1"],
        model_prefix="openai/",
    ),
    ProviderSpec(
        name="deepseek",
        display_name="DeepSeek",
        api_base="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_models=["deepseek-chat", "deepseek-reasoner"],
        model_prefix="deepseek/",
    ),
    ProviderSpec(
        name="groq",
        display_name="Groq",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_models=["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        model_prefix="groq/",
    ),
    ProviderSpec(
        name="gemini",
        display_name="Google Gemini",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        default_models=["gemini-2.5-pro", "gemini-2.5-flash"],
        model_prefix="gemini/",
    ),
    ProviderSpec(
        name="qwen",
        display_name="Qwen (DashScope)",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_models=["qwen-max", "qwen-turbo"],
        model_prefix="qwen/",
    ),
    ProviderSpec(
        name="minimax",
        display_name="MiniMax",
        api_base="https://api.minimax.chat/v1",
        api_key_env="MINIMAX_API_KEY",
        default_models=["abab6.5s-chat"],
        model_prefix="minimax/",
    ),
    ProviderSpec(
        name="moonshot",
        display_name="Moonshot / Kimi",
        api_base="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        default_models=["moonshot-v1-128k"],
        model_prefix="moonshot/",
    ),
    ProviderSpec(
        name="ollama_cloud",
        display_name="Ollama (Cloud)",
        api_base="https://ollama.com/v1",
        api_key_env="OLLAMA_API_KEY",
        default_models=[
            "llama3.3",
            "qwen2.5",
            "deepseek-r1",
            "mistral",
            "gemma2",
            "phi4",
        ],
        model_prefix="ollama_cloud/",
    ),
    # Local providers
    ProviderSpec(
        name="ollama",
        display_name="Ollama (Local)",
        api_base="",
        api_key_env="",
        default_models=["llama3.2", "qwen2.5", "mistral"],
        model_prefix="",
    ),
    ProviderSpec(
        name="llamacpp",
        display_name="Llama.cpp (Local)",
        api_base="http://localhost:8080/v1",
        api_key_env="",
        default_models=[],
        model_prefix="",
    ),
    ProviderSpec(
        name="lmstudio",
        display_name="LM Studio (Local)",
        api_base="",
        api_key_env="",
        default_models=["llama-3.2-3b-instruct", "qwen2.5-7b-instruct"],
        model_prefix="",
    ),
    ProviderSpec(
        name="vllm",
        display_name="vLLM (Local)",
        api_base="http://localhost:8000/v1",
        api_key_env="",
        default_models=[],
        model_prefix="vllm/",
    ),
    # Kept for backward compatibility (not shown in onboarding)
    ProviderSpec(
        name="zhipu",
        display_name="Zhipu AI",
        api_base="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        default_models=["glm-4", "glm-4-flash"],
        model_prefix="zhipu/",
    ),
)

_SPEC_BY_NAME: dict[str, ProviderSpec] = {s.name: s for s in PROVIDERS}
_SPEC_BY_PREFIX: dict[str, ProviderSpec] = {s.model_prefix: s for s in PROVIDERS if s.model_prefix}


class ProviderRegistry:
    """Lookup provider metadata and resolve model strings to provider + model."""

    @staticmethod
    def get_spec(name: str) -> ProviderSpec | None:
        return _SPEC_BY_NAME.get(name)

    @staticmethod
    def resolve_model(model_string: str, *, provider: str = "") -> tuple[ProviderSpec, str]:
        """Parse 'provider/model' into (ProviderSpec, bare_model_name).

        When ``provider`` is set (non-empty), it takes priority over prefix
        detection.  This allows model strings like ``openai/gpt-oss-120b`` to
        route through OpenRouter when ``provider='openrouter'``.

        If model_string has no known prefix and no explicit provider,
        falls back to OpenRouter.

        Examples:
            resolve_model('anthropic/claude-sonnet-4')
                -> (anthropic_spec, 'claude-sonnet-4')
            resolve_model('openai/gpt-oss-120b', provider='openrouter')
                -> (openrouter_spec, 'openai/gpt-oss-120b')
            resolve_model('gpt-4o')
                -> (openrouter_spec, 'gpt-4o')
            resolve_model('ollama/llama3.2')
                -> (ollama_spec, 'llama3.2')
        """
        if provider:
            spec = _SPEC_BY_NAME.get(provider)
            if spec:
                # Strip the provider's own prefix if present in the model string
                if spec.model_prefix and model_string.startswith(spec.model_prefix):
                    bare_model = model_string[len(spec.model_prefix) :]
                else:
                    bare_model = model_string
                return spec, bare_model
            logger.warning(
                "Explicit provider '{}' not found in registry, falling back to prefix detection",
                provider,
            )

        for prefix, spec in _SPEC_BY_PREFIX.items():
            if model_string.startswith(prefix):
                bare_model = model_string[len(prefix) :]
                return spec, bare_model

        for spec in PROVIDERS:
            if model_string.startswith(spec.name + "/"):
                bare_model = model_string[len(spec.name) + 1 :]
                return spec, bare_model

        openrouter = _SPEC_BY_NAME.get("openrouter")
        if openrouter:
            return openrouter, model_string

        raise ValueError(f"Cannot resolve provider for model: {model_string}")

    @staticmethod
    def list_providers() -> list[ProviderSpec]:
        return list(PROVIDERS)


def _get_api_key(spec: ProviderSpec, providers_config: dict[str, ProviderEntry]) -> str:
    """Resolve API key from config, then fall back to environment variable."""
    import os

    entry = providers_config.get(spec.name)
    if entry and entry.api_key:
        return entry.api_key

    if spec.api_key_env:
        return os.environ.get(spec.api_key_env, "")

    return ""


def _get_api_base(spec: ProviderSpec, providers_config: dict[str, ProviderEntry]) -> str:
    """Resolve API base URL from config, falling back to spec default."""
    entry = providers_config.get(spec.name)
    if entry and entry.api_base:
        return entry.api_base
    return spec.api_base


def create_provider(config: GripConfig) -> LLMProvider:
    """Instantiate the correct LLMProvider for the configured default model.

    Uses the explicit ``provider`` field (if set) or model string prefix to
    pick a provider, then creates either a LiteLLM-backed or direct
    OpenAI-compatible adapter.
    """
    model_string = config.agents.defaults.model
    explicit_provider = config.agents.defaults.provider
    spec, bare_model = ProviderRegistry.resolve_model(model_string, provider=explicit_provider)
    api_key = _get_api_key(spec, config.providers)
    api_base = _get_api_base(spec, config.providers)

    logger.info(
        "Creating provider: {} (model={}, base={})",
        spec.display_name,
        bare_model,
        api_base,
    )

    if spec.name in ("ollama", "ollama_cloud", "llamacpp", "lmstudio", "vllm") or not api_key:
        from grip.providers.openai_provider import OpenAICompatProvider

        return OpenAICompatProvider(
            provider_name=spec.display_name,
            api_base=api_base,
            api_key=api_key or "not-needed",
            default_model=bare_model,
        )

    from grip.providers.litellm_provider import LiteLLMProvider

    return LiteLLMProvider(
        provider_name=spec.display_name,
        model_prefix=spec.model_prefix.rstrip("/"),
        api_key=api_key,
        api_base=api_base,
        default_model=bare_model,
    )
