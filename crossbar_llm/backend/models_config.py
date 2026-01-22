MODELS_CONFIG = {
    "OpenRouter": [
        "Qwen/Qwen3-32B",
        "deepseek-v3-2-251201",
        "deepseek-r1-250528",
        "gemini-2.5-flash-nothinking",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4.5",
        "gpt-5.2",
        "gpt-oss-120b",
    ],
    "OpenAI": [
        "gpt-5.1",
        "gpt-5",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4o-mini",
        "o4-mini",
        "o3",
        "o3-mini",
        "o1",
        "gpt-4.1-nano",
    ],
    "Anthropic": [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-1",
    ],
    "Google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ],
    "Groq": [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "llama-3.1-8b-instant",
        "groq/compound",
        "groq/compound-mini",
    ],
    "Ollama": [
        "codestral:latest",
        "llama3:instruct",
        "tomasonjo/codestral-text2cypher:latest",
        "tomasonjo/llama3-text2cypher-demo:latest",
        "llama3.1:8b",
        "qwen2:7b-instruct",
        "gemma2:latest",
    ],
}


def get_all_models():
    """
    Get all model configurations.
    Returns a dictionary with provider names as keys and model lists as values.
    """
    return MODELS_CONFIG


def get_models_by_provider(provider: str):
    """
    Get models for a specific provider.
    
    Args:
        provider: The provider name (case-insensitive)
    
    Returns:
        List of model names for the provider, or empty list if not found
    """
    for key, models in MODELS_CONFIG.items():
        if key.lower() == provider.lower():
            return models
    return []


def get_all_model_names():
    """
    Get a flat list of all model names across all providers.
    
    Returns:
        List of all model names
    """
    all_models = []
    for models in MODELS_CONFIG.values():
        all_models.extend(models)
    return all_models


def get_provider_for_model_name(model_name: str):
    """
    Find which provider a model belongs to.
    
    Args:
        model_name: The model name to search for
    
    Returns:
        Provider name if found, None otherwise
    """
    for provider, models in MODELS_CONFIG.items():
        if model_name in models:
            return provider
    return None


def register_model(provider: str, model_name: str) -> bool:
    """
    Dynamically register a model to the configuration if it doesn't exist.
    
    Args:
        provider: The provider name (e.g., "OpenRouter", "OpenAI")
        model_name: The model name to register
    
    Returns:
        True if model was added, False if it already exists
    """
    # Normalize provider name (case-insensitive matching)
    matched_provider = None
    for key in MODELS_CONFIG.keys():
        if key.lower() == provider.lower():
            matched_provider = key
            break
    
    if matched_provider is None:
        # Create new provider entry
        MODELS_CONFIG[provider] = [model_name]
        return True
    
    if model_name not in MODELS_CONFIG[matched_provider]:
        MODELS_CONFIG[matched_provider].append(model_name)
        return True
    
    return False


def ensure_models_registered(provider: str, models: list) -> list:
    """
    Ensure all models in the list are registered under the specified provider.
    
    Args:
        provider: The provider name
        models: List of model names to register
    
    Returns:
        List of newly registered model names
    """
    newly_registered = []
    for model_name in models:
        if register_model(provider, model_name):
            newly_registered.append(model_name)
    return newly_registered
