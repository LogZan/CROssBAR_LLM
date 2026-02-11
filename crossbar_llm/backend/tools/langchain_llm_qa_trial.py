import json
import logging
import os
import sys
import threading
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from wrapt_timeout_decorator import *

# Import path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules for Neo4J connection and schema extraction
from typing import Literal, Union

import numpy as np
import pandas as pd
from .neo4j_query_corrector import correct_query, extract_cypher
from .neo4j_query_executor_extractor import Neo4jGraphHelper
from .entity_centric_schema_resolver import EntityCentricSchemaResolver
from .qa_templates import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_OUTPUT_PARSER_PROMPT,
    VECTOR_SEARCH_CYPHER_GENERATION_PROMPT,
    MULTI_HOP_DECISION_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama, Replicate
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Import the Language Model wrappers
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, validate_call

try:
    import tiktoken
except Exception:
    tiktoken = None


def _count_tokens(text: str, model_name: str = None) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return len(text.split())
    try:
        if model_name:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def configure_logging(verbose=False, log_filename="query_log.log"):
    """
    Configure logging for the application based on verbosity level.
    
    Args:
        verbose (bool): Whether to show detailed debug logs
        log_filename (str): Name of the log file to write logs to
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(parent_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    # Set up file handler for all logs
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Set up handlers based on verbosity
    handlers = [file_handler]
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        handlers=handlers,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Log configuration completed
    log_level = "DEBUG" if verbose else "INFO"
    logging.info(f"Logging initialized with level: {log_level}, output to: {log_path}")


class Config(BaseModel):
    """
    Config class for handling environment variables.
    It loads the variables from a .env file and makes them available as attributes.
    """

    load_dotenv()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "default")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "default")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "default")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "default")
    replicate_api_key: str = os.getenv("REPLICATE_API_KEY", "default")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "default")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "default")
    neo4j_usr: str = os.getenv("NEO4J_USERNAME")
    neo4j_password: str = os.getenv("MY_NEO4J_PASSWORD")
    neo4j_db_name: str = os.getenv("NEO4J_DATABASE_NAME")
    neo4j_uri: str = os.getenv("NEO4J_URI")


class Neo4JConnection:
    """
    Neo4JConnection class to handle interactions with a Neo4J database.
    It encapsulates the connection details and provides methods to interact with the database.
    """

    def __init__(
        self,
        user: str,
        password: str,
        db_name: str,
        uri: str,
        reset_schema: bool = False,
        create_vector_indexes: bool = False,
    ):

        self.graph_helper = Neo4jGraphHelper(
            URI=uri,
            user=user,
            password=password,
            db_name=db_name,
            reset_schema=reset_schema,
            create_vector_indexes=create_vector_indexes,
        )

        self.schema = self.graph_helper.create_graph_schema_variables()

    @validate_call
    def execute_query(self, query: str, top_k: int = 6) -> list:
        """
        Method to execute a given Cypher query against the Neo4J database.
        It returns the top k results.
        """
        return self.graph_helper.execute(query, top_k)


class OpenAILanguageModel:
    """
    OpenAILanguageModel class for interacting with OpenAI's language models.
    It initializes the model with given API key and specified parameters.
    """

    def __init__(
        self, api_key: str, model_name: str = None, temperature: float | int = None
    ):
        self.model_name = model_name or "gpt-3.5-turbo-instruct"
        self.temperature = temperature or 0
        
        # Models that don't support temperature parameter
        no_temp_models = [
            "gpt-5.1",
            "gpt-5",
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-4.1",
            "o4-mini",
            "o3",
            "o3-mini",
            "o1",
            "o1-mini",
            "o1-pro",
        ]
        
        if self.model_name in no_temp_models:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model_name=self.model_name,
                request_timeout=30,
                temperature=1,
            )
        else:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                request_timeout=30,
            )


class GoogleGenerativeLanguageModel:
    """
    GoogleGenerativeLanguageModel class for interacting with Google's Generative AI models.
    It initializes the model with given API key and specified parameters.
    """

    def __init__(
        self, api_key: str, model_name: str = None, temperature: float | int = None
    ):
        self.model_name = model_name or "gemini-1.5-pro-latest"
        self.temperature = temperature or 0
        self.llm = GoogleGenerativeAI(
            api_key=api_key,
            model=self.model_name,
            temperature=self.temperature,
            request_timeout=120,
        )


class AnthropicLanguageModel:
    """
    AnthropicLanguageModel class for interacting with Anthropic's language models.
    It initializes the model with given API key and specified parameters.
    """

    def __init__(
        self, api_key: str, model_name: str = None, temperature: float | int = None
    ):
        self.model_name = model_name or "claude-3-opus-20240229"
        self.temperature = temperature or 0
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        )


class GroqLanguageModel:
    """
    GroqLanguageModel class for interacting with Groq's language models.
    It initializes the model with given API key and specified parameters.
    """

    def __init__(
        self, api_key: str, model_name: str = None, temperature: float | int = None
    ):
        self.model_name = model_name or "llama3-70b-8192"
        self.temperature = temperature or 0
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        )


class ReplicateLanguageModel:
    """
    ReplicateLanguageModel class for interacting with Replicate's language models.
    It initializes the model with given API key and specified parameters.
    """

    def __init__(
        self, api_key: str, model_name: str = None, temperature: float | int = None
    ):
        self.model_name = model_name or "replicate-1.0"
        self.temperature = temperature or 0
        self.llm = Replicate(
            replicate_api_key=api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        )


class OllamaLanguageModel:
    """
    OllamaLanguageModel class for interacting with Ollama's language models.
    """

    def __init__(self, model_name: str = None, temperature: float | int = None):
        self.model_name = model_name or "codestral:latest"
        self.temperature = temperature or 0
        self.llm = Ollama(
            model=self.model_name, temperature=self.temperature, timeout=30
        )


class NVIDIALanguageModel:
    """
    NVIDIALanguageModel class for interacting with NVIDIA's language models.
    It initializes the model with given API key and specified parameters.
    """

    def __init__(
        self, api_key: str, model_name: str = None, temperature: float | int = None
    ):
        self.model_name = model_name or "meta/llama-3.1-405b-instruct"
        self.temperature = temperature or 0
        self.llm = ChatNVIDIA(
            api_key=api_key, model=self.model_name, temperature=self.temperature
        )


class OpenRouterLanguageModel:
    """
    OpenRouterLanguageModel class for interacting with OpenRouter's language models.
    It initializes the model with given API key and specified parameters.
    """

    def __init__(
        self, api_key: str, model_name: str = None, temperature: float | int = None,
        base_url: str = None
    ):
        self.model_name = model_name or "deepseek/deepseek-r1"
        self.temperature = temperature or 0
        self.base_url = base_url or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.llm = ChatOpenAI(
            openai_api_key=api_key, model_name=self.model_name, temperature=self.temperature, request_timeout=180, openai_api_base=self.base_url
        )


class QueryChain:
    """
    QueryChain class to handle the generation, correction, and parsing of Cypher queries using language models.
    It encapsulates the entire process as a single chain of operations.
    """

    def __init__(
        self,
        cypher_llm: Union[
            OpenAILanguageModel,
            GoogleGenerativeLanguageModel,
            AnthropicLanguageModel,
            GroqLanguageModel,
            ReplicateLanguageModel,
            OllamaLanguageModel,
            NVIDIALanguageModel,
            OpenRouterLanguageModel,
        ],
        qa_llm: Union[
            OpenAILanguageModel,
            GoogleGenerativeLanguageModel,
            AnthropicLanguageModel,
            GroqLanguageModel,
            ReplicateLanguageModel,
            OllamaLanguageModel,
            NVIDIALanguageModel,
            OpenRouterLanguageModel,
        ],
        schema: dict,
        verbose: bool = False,
        search_type: Literal["vector_search", "db_search"] = "db_search",
        use_entity_centric_resolver: bool = False,
        resolver_config: dict = None,
        neo4j_helper: Neo4jGraphHelper = None,
    ):

        if search_type == "db_search":
            self.cypher_chain = CYPHER_GENERATION_PROMPT | cypher_llm | StrOutputParser()
        else:
            self.cypher_chain = VECTOR_SEARCH_CYPHER_GENERATION_PROMPT | cypher_llm | StrOutputParser()

        self.qa_chain = CYPHER_OUTPUT_PARSER_PROMPT | qa_llm | StrOutputParser()
        self.schema = schema
        self.verbose = verbose
        self.search_type = search_type
        self.last_cypher_prompt_tokens = 0
        self.last_cypher_output_tokens = 0
        self.last_qa_prompt_tokens = 0
        self.last_qa_output_tokens = 0

        self.generated_queries = []
        self.last_resolution_used = False
        self.last_resolution_reason = "not_attempted"

        # Initialize EntityCentricSchemaResolver if enabled
        self.resolver = None
        if use_entity_centric_resolver and neo4j_helper:
            try:
                self.resolver = EntityCentricSchemaResolver(
                    llm_client=cypher_llm,
                    neo4j_helper=neo4j_helper,
                    cache_dir=resolver_config.get("cache_dir", "cache") if resolver_config else "cache",
                    config=resolver_config or {}
                )
                logging.info("EntityCentricSchemaResolver initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize EntityCentricSchemaResolver: {e}")
                self.resolver = None

    @timeout(180)
    @validate_call
    def run_cypher_chain(
        self,
        question: str,
        vector_index: str = None,
        embedding: Union[list[float], None] = None,
    ) -> str:
        """
        Executes the query chain: generates a query, corrects it, and returns the corrected query.
        """

        # Try entity-centric schema resolution
        schema_context = self.schema  # Default to full schema
        if self.resolver:
            try:
                logging.info("Attempting entity-centric schema resolution")
                resolved_schema = self.resolver.resolve(question)
                if resolved_schema:
                    self.last_resolution_used = True
                    self.last_resolution_reason = "resolved"
                    self.last_resolution_detail = None
                    schema_context = resolved_schema
                    logging.info("Using entity-centric filtered schema")
                else:
                    self.last_resolution_used = False
                    self.last_resolution_reason = "fallback_full_schema"
                    self.last_resolution_detail = getattr(self.resolver, "last_failure_reason", "unknown")
                    logging.info("Entity resolution returned None, using full schema")
            except Exception as e:
                self.last_resolution_used = False
                self.last_resolution_reason = "resolver_error"
                self.last_resolution_detail = "resolver_exception"
                logging.warning(f"Entity resolution failed: {e}, using full schema")
                schema_context = self.schema
        else:
            self.last_resolution_used = False
            self.last_resolution_reason = "resolver_disabled"
            self.last_resolution_detail = "resolver_disabled"

        if self.search_type == "db_search":
            anchor_entities = self._format_anchor_entities(schema_context)
            resolved_schema = json.dumps(schema_context, ensure_ascii=False, indent=2)
            logging.getLogger("batch_pipeline").info("Cypher gen: building prompt (db_search)")
            prompt_text = CYPHER_GENERATION_PROMPT.format(
                node_types=schema_context["nodes"],
                node_properties=schema_context["node_properties"],
                edge_properties=schema_context["edge_properties"],
                edges=schema_context["edges"],
                resolved_schema=resolved_schema,
                anchor_entities=anchor_entities,
                question=question,
            )
            self.last_cypher_prompt_tokens = _count_tokens(prompt_text)
            logging.getLogger("batch_pipeline").info("Cypher gen: invoking LLM")
            self.generated_query = (
                self.cypher_chain.invoke(
                    {
                        "node_types": schema_context["nodes"],
                        "node_properties": schema_context["node_properties"],
                        "edge_properties": schema_context["edge_properties"],
                        "edges": schema_context["edges"],
                        "resolved_schema": resolved_schema,
                        "anchor_entities": anchor_entities,
                        "question": question,
                    }
                )
                .strip()
                .strip("\n")
                .replace("cypher", "")
                .strip("`")
                .replace("''", "'")
                .replace('""', '"')
            )
            logging.getLogger("batch_pipeline").info("Cypher gen: raw LLM output received")
            self.generated_query = self._validate_or_retry_query(
                self.generated_query, schema_context, prompt_text, question
            )

        elif self.search_type == "vector_search" and embedding is None:
            anchor_entities = self._format_anchor_entities(schema_context)
            resolved_schema = json.dumps(schema_context, ensure_ascii=False, indent=2)
            logging.getLogger("batch_pipeline").info("Cypher gen: building prompt (vector_search)")
            prompt_text = VECTOR_SEARCH_CYPHER_GENERATION_PROMPT.format(
                vector_index=vector_index,
                node_types=schema_context["nodes"],
                node_properties=schema_context["node_properties"],
                edge_properties=schema_context["edge_properties"],
                edges=schema_context["edges"],
                resolved_schema=resolved_schema,
                anchor_entities=anchor_entities,
                question=question,
            )
            self.last_cypher_prompt_tokens = _count_tokens(prompt_text)
            logging.getLogger("batch_pipeline").info("Cypher gen: invoking LLM")
            self.generated_query = (
                self.cypher_chain.invoke(
                    {
                        "node_types": schema_context["nodes"],
                        "node_properties": schema_context["node_properties"],
                        "edge_properties": schema_context["edge_properties"],
                        "edges": schema_context["edges"],
                        "resolved_schema": resolved_schema,
                        "anchor_entities": anchor_entities,
                        "question": question,
                        "vector_index": vector_index,
                    }
                )
                .strip()
                .strip("\n")
                .replace("cypher", "")
                .strip("`")
                .replace("''", "'")
                .replace('""', '"')
            )
            logging.getLogger("batch_pipeline").info("Cypher gen: raw LLM output received")
            self.generated_query = self._validate_or_retry_query(
                self.generated_query, schema_context, prompt_text, question
            )

        elif self.search_type == "vector_search" and embedding is not None:
            anchor_entities = self._format_anchor_entities(schema_context)
            resolved_schema = json.dumps(schema_context, ensure_ascii=False, indent=2)
            logging.getLogger("batch_pipeline").info("Cypher gen: building prompt (vector_search + embedding)")
            prompt_text = VECTOR_SEARCH_CYPHER_GENERATION_PROMPT.format(
                vector_index=vector_index,
                node_types=schema_context["nodes"],
                node_properties=schema_context["node_properties"],
                edge_properties=schema_context["edge_properties"],
                edges=schema_context["edges"],
                resolved_schema=resolved_schema,
                anchor_entities=anchor_entities,
                question=question,
            )
            self.last_cypher_prompt_tokens = _count_tokens(prompt_text)

            logging.getLogger("batch_pipeline").info("Cypher gen: invoking LLM")
            self.generated_query = (
                self.cypher_chain.invoke(
                    {
                        "node_types": schema_context["nodes"],
                        "node_properties": schema_context["node_properties"],
                        "edge_properties": schema_context["edge_properties"],
                        "edges": schema_context["edges"],
                        "resolved_schema": resolved_schema,
                        "anchor_entities": anchor_entities,
                        "question": question,
                        "vector_index": vector_index,
                    }
                )
                .strip()
                .strip("\n")
                .replace("cypher", "")
                .strip("`")
                .replace("''", "'")
                .replace('""', '"')
            )
            logging.getLogger("batch_pipeline").info("Cypher gen: raw LLM output received")
            self.generated_query = self._validate_or_retry_query(
                self.generated_query, schema_context, prompt_text, question
            )

            self.generated_query = self.generated_query.format(user_input=embedding)

        # Basic validation to check if it looks like a Cypher query
        valid_prefixes = ["MATCH", "CREATE", "MERGE", "CALL", "RETURN", "WITH", "UNWIND"]
        is_query = any(self.generated_query.upper().lstrip().startswith(prefix) for prefix in valid_prefixes)
        
        if not is_query:
            logging.warning(f"Generated text does not appear to be a Cypher query: {self.generated_query}")
            return self.generated_query  # Return the text as is, assuming it's an error message

        self.last_cypher_output_tokens = _count_tokens(self.generated_query)

        logging.getLogger("batch_pipeline").info("Cypher gen: correcting query")
        corrected_query = correct_query(
            query=self.generated_query, edge_schema=schema_context
        )

        logging.getLogger("batch_pipeline").info("Cypher gen: enforcing anchor")
        enforced_query = self._enforce_anchor_query(corrected_query, schema_context)
        if enforced_query != corrected_query:
            logging.info("Anchor enforcement applied to Cypher query")
            corrected_query = enforced_query

        self.generated_queries.append(corrected_query)

        # Logging generated and corrected queries
        logging.info(f"Generated Query: {self.generated_query}")
        logging.info(f"Corrected Query: {corrected_query}")

        return corrected_query

    def _format_anchor_entities(self, schema_context: dict) -> str:
        anchors = schema_context.get("anchor_entities") or []
        if not anchors:
            return ""
        lines = []
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            anchor_id = anchor.get("id") or anchor.get("node_id")
            anchor_type = anchor.get("type") or anchor.get("node_type")
            name = anchor.get("name") or anchor.get("identifier")
            if anchor_id and anchor_type:
                if name:
                    lines.append(f"- {anchor_type} id={anchor_id} name={name}")
                else:
                    lines.append(f"- {anchor_type} id={anchor_id}")
        return "\n".join(lines)

    def _validate_or_retry_query(self, query: str, schema_context: dict, prompt_text: str, question: str) -> str:
        query = self._sanitize_cypher_text(query)
        invalid = self._find_invalid_headers(query, schema_context)
        if not invalid:
            return query

        logging.warning(f"Invalid headers detected: {invalid}")
        resolved_schema = json.dumps(schema_context, ensure_ascii=False, indent=2)
        retry_prompt = (
            f"{prompt_text}\n\n"
            f"Your previous query used headers not present in the resolved schema: {sorted(invalid)}.\n"
            "Regenerate using only allowed headers from the resolved schema."
        )
        self.last_cypher_prompt_tokens = _count_tokens(retry_prompt)
        retry_query = (
            self.cypher_chain.invoke(
                {
                    "node_types": schema_context["nodes"],
                    "node_properties": schema_context["node_properties"],
                    "edge_properties": schema_context["edge_properties"],
                    "edges": schema_context["edges"],
                    "resolved_schema": resolved_schema,
                    "anchor_entities": self._format_anchor_entities(schema_context),
                    "question": question,
                }
            )
            .strip()
            .strip("\n")
            .replace("cypher", "")
            .strip("`")
        )
        retry_query = self._sanitize_cypher_text(retry_query)
        invalid_retry = self._find_invalid_headers(retry_query, schema_context)
        if invalid_retry:
            logging.warning(f"Retry still has invalid headers: {invalid_retry}")
        return retry_query

    def _sanitize_cypher_text(self, text: str) -> str:
        if not text:
            return text
        cleaned = extract_cypher(text.strip())
        lines = cleaned.splitlines()
        for i, line in enumerate(lines):
            if re.match(r"\\s*(MATCH|CREATE|MERGE|CALL|RETURN|WITH|UNWIND)\\b", line, re.IGNORECASE):
                return "\n".join(lines[i:]).strip()
        # Fallback: strip leading comment/blank lines
        while lines and (not lines[0].strip() or lines[0].lstrip().startswith("//")):
            lines.pop(0)
        return "\n".join(lines).strip()

    def _find_invalid_headers(self, query: str, schema_context: dict) -> set[str]:
        invalid = set()
        node_props, edge_props = self._build_allowed_headers(schema_context)
        try:
            var_types = self._extract_var_types(query)
            used = self._extract_used_headers(query)
        except re.error as e:
            logging.warning(f"Header validation regex error: {e}")
            return invalid
        for var_name, prop in used:
            var_type = var_types.get(var_name)
            if var_type == "edge":
                if prop not in edge_props:
                    invalid.add(prop)
            elif var_type == "node":
                if prop not in node_props:
                    invalid.add(prop)
            else:
                if prop not in node_props and prop not in edge_props:
                    invalid.add(prop)
        return invalid

    def _build_allowed_headers(self, schema_context: dict) -> tuple[set[str], set[str]]:
        node_allowed = set()
        edge_allowed = set()
        for entry in schema_context.get("node_properties", []) or []:
            for prop in entry.get("properties", []) or []:
                prop_name = prop.get("property") if isinstance(prop, dict) else prop
                if prop_name:
                    node_allowed.add(prop_name)
        for entry in schema_context.get("edge_properties", []) or []:
            for prop in entry.get("properties", []) or []:
                prop_name = prop.get("property") if isinstance(prop, dict) else prop
                if prop_name:
                    edge_allowed.add(prop_name)
        return node_allowed, edge_allowed

    @staticmethod
    def _extract_used_headers(query: str) -> set[tuple[str, str]]:
        used = set()
        pattern = r"\b(\w+)\.([A-Za-z_][A-Za-z0-9_]*)\b"
        for match in re.finditer(pattern, query):
            used.add((match.group(1), match.group(2)))
        return used

    @staticmethod
    def _extract_var_types(query: str) -> dict[str, str]:
        var_types: dict[str, str] = {}
        node_pattern = r"\(\s*(\w+)\s*:\s*\w+"
        edge_pattern = r"\[\s*(\w+)\s*:\s*\w+"
        for match in re.finditer(node_pattern, query):
            var_types[match.group(1)] = "node"
        for match in re.finditer(edge_pattern, query):
            var_types[match.group(1)] = "edge"
        return var_types

    def _enforce_anchor_query(self, query: str, schema_context: dict) -> str:
        anchors = schema_context.get("anchor_entities") or []
        if not anchors:
            return query

        anchor = anchors[0]
        anchor_id = anchor.get("id")
        anchor_type = anchor.get("type")
        if not anchor_id or not anchor_type:
            return query

        if anchor_id in query:
            return query

        # Try to inject id into a node pattern with matching label
        label_pattern_str = rf"\(\s*(?P<var>\w+)\s*:\s*{re.escape(anchor_type)}\s*(?P<props>\{{[^}}]*\}})?\s*\)"
        try:
            label_pattern = re.compile(label_pattern_str)
        except re.error as e:
            logging.warning(
                f"Anchor enforce regex compile failed: {e} anchor_type={anchor_type} pattern={label_pattern_str}"
            )
            return query
        match = label_pattern.search(query)
        if match:
            var_name = match.group("var")
            props = match.group("props")
            if props:
                if "id" in props:
                    return query
                injected = props[:-1] + f', id:\"{anchor_id}\"' + "}"
            else:
                injected = f'{{id:\"{anchor_id}\"}}'
            replacement = f"({var_name}:{anchor_type} {injected})"
            return query[: match.start()] + replacement + query[match.end():]

        # Fallback: replace the first node pattern with anchor label + id
        generic_pattern = re.compile(
            r"\(\s*(?P<var>\w+)(?:\s*:\s*(?P<label>\w+))?\s*(?P<props>\{[^}]*\})?\s*\)"
        )
        match = generic_pattern.search(query)
        if not match:
            return f'MATCH (p:{anchor_type} {{id:\"{anchor_id}\"}})\\nRETURN p'

        var_name = match.group("var")
        props = match.group("props")
        if props:
            if "id" in props:
                injected = props
            else:
                injected = props[:-1] + f', id:\"{anchor_id}\"' + "}"
        else:
            injected = f'{{id:\"{anchor_id}\"}}'
        replacement = f"({var_name}:{anchor_type} {injected})"
        return query[: match.start()] + replacement + query[match.end():]


class MultiHopReasoner:
    """
    Multi-hop reasoning engine for knowledge graph exploration.

    At each step the LLM chooses one of four actions:
      A. CONTINUE – keep exploring the current node (different properties / edges).
      B. JUMP     – move to a different node (type + identifier supplied by LLM).
      C. ANSWER   – stop and produce the final answer.
      D. OVERVIEW – run a global / cross-node query.

    The accumulated evidence and conversation context are carried forward
    across hops so the LLM can make informed decisions.
    """

    def __init__(
        self,
        llm,
        neo4j_connection,
        query_chain_factory,
        max_steps: int = 5,
        search_type: str = "db_search",
    ):
        self.llm = llm
        self.neo4j_connection = neo4j_connection
        self.query_chain_factory = query_chain_factory
        self.max_steps = max_steps
        self.search_type = search_type
        self.decision_chain = MULTI_HOP_DECISION_PROMPT | llm | StrOutputParser()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_decision(raw_text: str) -> dict:
        """Parse the LLM decision JSON, tolerating minor formatting issues."""
        try:
            return json.loads(raw_text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {"action": "C", "reason": "Failed to parse decision, defaulting to ANSWER"}

    @staticmethod
    def _summarize_evidence(evidence: list) -> str:
        if not evidence:
            return "No evidence collected yet."
        try:
            return json.dumps(evidence[:10], ensure_ascii=False, indent=1)
        except Exception:
            return str(evidence[:10])

    # ------------------------------------------------------------------
    # core loop
    # ------------------------------------------------------------------
    def run(self, question: str, top_k: int = 5) -> dict:
        """
        Execute multi-hop reasoning.

        Returns a dict with keys:
            - evidence   : list of all collected KG results
            - trace      : list[dict] per-step trace
            - final_action : the terminal action letter
        """
        evidence: list = []
        trace: list = []
        current_node: str = "Not yet determined (initial step)"

        for step in range(1, self.max_steps + 1):
            # --- 1. ask the LLM what to do ---
            decision_raw = self.decision_chain.invoke({
                "question": question,
                "current_node": current_node,
                "evidence": self._summarize_evidence(evidence),
                "step": str(step),
                "max_steps": str(self.max_steps),
            })
            decision = self._parse_decision(decision_raw)
            action = decision.get("action", "C").upper().strip()
            reason = decision.get("reason", "")

            logging.info(
                f"MultiHopReasoner step {step}/{self.max_steps}: "
                f"action={action} reason={reason}"
            )

            step_record = {
                "step": step,
                "action": action,
                "reason": reason,
                "decision_raw": decision_raw,
            }

            # --- 2. act on the decision ---
            if action == "C":
                # ANSWER – stop
                step_record["status"] = "terminate"
                trace.append(step_record)
                break

            if action == "B":
                # JUMP to a different node
                target = decision.get("jump_target") or {}
                node_type = target.get("node_type", "")
                identifier = target.get("identifier", "")
                if node_type and identifier:
                    current_node = f"{node_type}: {identifier}"
                    jump_question = (
                        f"{question}\n\n"
                        f"Focus on {node_type} with identifier '{identifier}'."
                    )
                else:
                    jump_question = question

                query, result = self._query_kg(jump_question, top_k)
                step_record.update({
                    "cypher": query,
                    "result_count": len(result) if isinstance(result, list) else (0 if not result else 1),
                    "status": "jump",
                    "jump_target": {"node_type": node_type, "identifier": identifier},
                })
                if isinstance(result, list):
                    evidence.extend(result)
                elif result:
                    evidence.append(result)

            elif action == "D":
                # OVERVIEW – global query
                hint = decision.get("overview_hint") or ""
                overview_question = (
                    f"{question}\n\n"
                    f"Provide a global overview. {hint}"
                ).strip()
                query, result = self._query_kg(overview_question, top_k)
                step_record.update({
                    "cypher": query,
                    "result_count": len(result) if isinstance(result, list) else (0 if not result else 1),
                    "status": "overview",
                })
                if isinstance(result, list):
                    evidence.extend(result)
                elif result:
                    evidence.append(result)

            else:
                # A  CONTINUE on the same node
                hint = decision.get("focus_hint") or ""
                continue_question = (
                    f"{question}\n\n"
                    f"Continue exploring: {current_node}. {hint}"
                ).strip()
                query, result = self._query_kg(continue_question, top_k)
                step_record.update({
                    "cypher": query,
                    "result_count": len(result) if isinstance(result, list) else (0 if not result else 1),
                    "status": "continue",
                })
                if isinstance(result, list):
                    evidence.extend(result)
                elif result:
                    evidence.append(result)

            trace.append(step_record)

        return {
            "evidence": evidence,
            "trace": trace,
            "final_action": action if 'action' in dir() else "C",
        }

    # ------------------------------------------------------------------
    def _query_kg(self, question: str, top_k: int) -> tuple:
        """Generate a Cypher query and execute it, returning (cypher, result)."""
        query_chain = self.query_chain_factory()
        try:
            cypher = query_chain.run_cypher_chain(question)
        except Exception as e:
            logging.warning(f"MultiHopReasoner cypher generation failed: {e}")
            return "", []
        try:
            result = self.neo4j_connection.execute_query(cypher, top_k=top_k)
        except Exception as e:
            logging.warning(f"MultiHopReasoner query execution failed: {e}")
            return cypher, []
        return cypher, result if result else []


class RunPipeline:

    def __init__(
        self,
        model_name=Union[
            str, list[str], dict[Literal["cypher_llm_model", "qa_llm_model"], str]
        ],
        verbose: bool = False,
        top_k: int = 5,
        reset_schema: bool = False,
        search_type: Literal["vector_search", "db_search"] = "db_search",
        use_entity_centric_resolver: bool = None,
        resolver_config: dict = None,
    ):

        self.verbose = verbose
        self.top_k = top_k
        self._resolver_status = threading.local()
        self._token_stats = threading.local()
        self.config: Config = Config()
        self.neo4j_connection: Neo4JConnection = Neo4JConnection(
            self.config.neo4j_usr,
            self.config.neo4j_password,
            self.config.neo4j_db_name,
            self.config.neo4j_uri,
            reset_schema=reset_schema,
            create_vector_indexes=False if search_type == "db_search" else True,
        )
        self.search_type = search_type

        # Load entity-centric resolver configuration
        if use_entity_centric_resolver is None or resolver_config is None:
            # Try to load from batch_config.yaml
            try:
                import yaml
                config_path = None
                # Walk up to find project root config/batch_config.yaml
                current = Path(__file__).resolve()
                for parent in current.parents:
                    candidate = parent / "config" / "batch_config.yaml"
                    if candidate.exists():
                        config_path = candidate
                        break
                if config_path and config_path.exists():
                    with open(config_path, 'r') as f:
                        batch_config = yaml.safe_load(f)
                        resolver_settings = batch_config.get("entity_centric_resolver", {})
                        if use_entity_centric_resolver is None:
                            use_entity_centric_resolver = resolver_settings.get("enabled", False)
                        if resolver_config is None:
                            resolver_config = resolver_settings
                        logging.info(f"Loaded entity_centric_resolver config from {config_path}")
                else:
                    logging.warning("batch_config.yaml not found when loading resolver config")
                    use_entity_centric_resolver = False
                    resolver_config = {}
            except Exception as e:
                logging.warning(f"Failed to load resolver config from batch_config.yaml: {e}")
                use_entity_centric_resolver = False
                resolver_config = {}

        self.use_entity_centric_resolver = use_entity_centric_resolver
        self.resolver_config = resolver_config
        logging.info(f"Entity-centric resolver enabled: {self.use_entity_centric_resolver}")

        # define llm type(s)
        self.define_llm(model_name)

        # define outputs list
        self.outputs = []

    def get_last_resolver_status(self) -> dict:
        return {
            "enabled": getattr(self._resolver_status, "enabled", False),
            "used": getattr(self._resolver_status, "used", False),
            "reason": getattr(self._resolver_status, "reason", "unknown"),
            "detail": getattr(self._resolver_status, "detail", None),
        }

    def get_last_token_stats(self) -> dict:
        return {
            "cypher_prompt_tokens": getattr(self._token_stats, "cypher_prompt_tokens", 0),
            "cypher_output_tokens": getattr(self._token_stats, "cypher_output_tokens", 0),
            "qa_prompt_tokens": getattr(self._token_stats, "qa_prompt_tokens", 0),
            "qa_output_tokens": getattr(self._token_stats, "qa_output_tokens", 0),
        }

    def define_llm(self, model_name):
        from models_config import get_provider_for_model_name
    
        provider_model_map = {
            "OpenAI": (OpenAILanguageModel, self.config.openai_api_key),
            "Google": (GoogleGenerativeLanguageModel, self.config.gemini_api_key),
            "Anthropic": (AnthropicLanguageModel, self.config.anthropic_api_key),
            "Groq": (GroqLanguageModel, self.config.groq_api_key),
            "Ollama": (OllamaLanguageModel, None),  # Ollama doesn't need an API key
            "Nvidia": (NVIDIALanguageModel, self.config.nvidia_api_key),
            "OpenRouter": (OpenRouterLanguageModel, self.config.openrouter_api_key),
        }
        
        def get_llm_for_model(model_name_str):
            """Helper function to get the appropriate LLM instance for a model name."""
            provider = get_provider_for_model_name(model_name_str)
            if not provider:
                raise ValueError(f"Unsupported Language Model Name: {model_name_str}")
            
            if provider not in provider_model_map:
                raise ValueError(f"Unsupported Provider: {provider}")
            
            model_class, api_key = provider_model_map[provider]
            
            # Ollama doesn't use an API key
            if provider == "Ollama":
                return model_class(model_name=model_name_str).llm
            else:
                return model_class(api_key, model_name=model_name_str).llm

        if isinstance(model_name, (dict, list)):

            if len(model_name) != 2:
                raise ValueError("Length of `model_name` must be 2")

            if isinstance(model_name, list):
                model_name = dict(zip(["cypher_llm_model", "qa_llm_model"], model_name))

            self.llm = {}
            for model_type, model_name_str in model_name.items():
                if model_type == "cypher_llm_model":
                    self.llm["cypher_llm"] = get_llm_for_model(model_name_str)
                elif model_type == "qa_llm_model":
                    self.llm["qa_llm"] = get_llm_for_model(model_name_str)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

        else:
            self.llm = get_llm_for_model(model_name)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def run_for_query(
        self,
        question: str,
        vector_index: str = None,
        embedding: Union[np.array, None] = None,
        reset_llm_type: bool = False,
        model_name: Union[
            str, list[str], dict[Literal["cypher_llm_model", "qa_llm_model"], str]
        ] = None,
        api_key: str = None,
    ) -> str:

        if api_key:
            self.config.openai_api_key = api_key
            self.config.gemini_api_key = api_key
            self.config.anthropic_api_key = api_key
            self.config.groq_api_key = api_key
            self.config.replicate_api_key = api_key
            self.config.nvidia_api_key = api_key
            self.config.openrouter_api_key = api_key

        else:
            self.config = Config()

        if reset_llm_type:
            self.define_llm(model_name=model_name)

        logging.info(
            f"Selected Language Model(s): {list(self.llm.values()) if isinstance(self.llm, dict) else self.llm}"
        )
        logging.info(f"Question: {question}")

        if isinstance(self.llm, dict):
            query_chain: QueryChain = QueryChain(
                cypher_llm=self.llm["cypher_llm"],
                qa_llm=self.llm["qa_llm"],
                schema=self.neo4j_connection.schema,
                search_type=self.search_type,
                use_entity_centric_resolver=self.use_entity_centric_resolver,
                resolver_config=self.resolver_config,
                neo4j_helper=self.neo4j_connection.graph_helper,
            )
        else:
            query_chain: QueryChain = QueryChain(
                cypher_llm=self.llm,
                qa_llm=self.llm,
                schema=self.neo4j_connection.schema,
                search_type=self.search_type,
                use_entity_centric_resolver=self.use_entity_centric_resolver,
                resolver_config=self.resolver_config,
                neo4j_helper=self.neo4j_connection.graph_helper,
            )

        if self.search_type == "db_search":
            corrected_query = query_chain.run_cypher_chain(question)
        else:
            corrected_query = query_chain.run_cypher_chain(
                question,
                vector_index=vector_index,
                embedding=self.handle_embedding(
                    embedding=embedding, vector_index=vector_index
                ),
            )

        self._resolver_status.enabled = bool(query_chain.resolver)
        self._resolver_status.used = bool(getattr(query_chain, "last_resolution_used", False))
        self._resolver_status.reason = getattr(query_chain, "last_resolution_reason", "unknown")
        self._resolver_status.detail = getattr(query_chain, "last_resolution_detail", None)
        self._token_stats.cypher_prompt_tokens = getattr(query_chain, "last_cypher_prompt_tokens", 0)
        self._token_stats.cypher_output_tokens = getattr(query_chain, "last_cypher_output_tokens", 0)

        return corrected_query

    def execute_query(
        self,
        query: str,
        question: str,
        model_name: str,
        reset_llm_type: bool = False,
        api_key: str = None,
    ) -> str:

        result = self.neo4j_connection.execute_query(query, top_k=self.top_k)

        if api_key:
            self.config.openai_api_key = api_key
            self.config.gemini_api_key = api_key
            self.config.anthropic_api_key = api_key
            self.config.groq_api_key = api_key
            self.config.replicate_api_key = api_key
            self.config.nvidia_api_key = api_key
            self.config.openrouter_api_key = api_key
        else:
            self.config = Config()

        if reset_llm_type:
            self.define_llm(model_name=model_name)

        logging.info(f"Query Result: {result}")

        if isinstance(self.llm, dict):
            query_chain: QueryChain = QueryChain(
                cypher_llm=self.llm["cypher_llm"],
                qa_llm=self.llm["qa_llm"],
                schema=self.neo4j_connection.schema,
                search_type=self.search_type,
                use_entity_centric_resolver=self.use_entity_centric_resolver,
                resolver_config=self.resolver_config,
                neo4j_helper=self.neo4j_connection.graph_helper,
            )
        else:
            query_chain: QueryChain = QueryChain(
                cypher_llm=self.llm,
                qa_llm=self.llm,
                schema=self.neo4j_connection.schema,
                search_type=self.search_type,
                use_entity_centric_resolver=self.use_entity_centric_resolver,
                resolver_config=self.resolver_config,
                neo4j_helper=self.neo4j_connection.graph_helper,
            )

        qa_prompt_text = CYPHER_OUTPUT_PARSER_PROMPT.format(
            output=result, input_question=question
        )
        self._token_stats.qa_prompt_tokens = _count_tokens(qa_prompt_text)

        final_output = query_chain.qa_chain.invoke(
            {"output": result, "input_question": question}
        ).strip("\n")
        self._token_stats.qa_output_tokens = _count_tokens(final_output)

        logging.info(f"{final_output}")

        return final_output, result

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def run_without_errors(
        self,
        question: str,
        vector_index: str = None,
        embedding: np.array = None,
        reset_llm_type: bool = False,
        model_name: Union[
            str, list[str], dict[Literal["cypher_llm_model", "qa_llm_model"], str]
        ] = None,
    ) -> str:

        if reset_llm_type:
            self.define_llm(model_name=model_name)

        logging.info(
            f"Selected Language Model(s): {list(self.llm.values()) if isinstance(self.llm, dict) else self.llm}"
        )
        logging.info(f"Question: {question}")

        if isinstance(self.llm, dict):
            query_chain: QueryChain = QueryChain(
                cypher_llm=self.llm["cypher_llm"],
                qa_llm=self.llm["qa_llm"],
                schema=self.neo4j_connection.schema,
                search_type=self.search_type,
                use_entity_centric_resolver=self.use_entity_centric_resolver,
                resolver_config=self.resolver_config,
                neo4j_helper=self.neo4j_connection.graph_helper,
            )
        else:
            query_chain: QueryChain = QueryChain(
                cypher_llm=self.llm,
                qa_llm=self.llm,
                schema=self.neo4j_connection.schema,
                search_type=self.search_type,
                use_entity_centric_resolver=self.use_entity_centric_resolver,
                resolver_config=self.resolver_config,
                neo4j_helper=self.neo4j_connection.graph_helper,
            )

        if self.search_type == "db_search":
            corrected_query = query_chain.run_cypher_chain(question)
        else:
            corrected_query = query_chain.run_cypher_chain(
                question,
                vector_index=vector_index,
                embedding=self.handle_embedding(
                    embedding=embedding, vector_index=vector_index
                ),
            )

        if not corrected_query:
            self.outputs.append((question, query_chain.generated_query, "", "", ""))
            return None

        try:
            result = self.neo4j_connection.execute_query(
                corrected_query, top_k=self.top_k
            )
            logging.info(f"Query Result: {result}")
        except Exception as e:
            logging.info(f"An error occurred trying to execute the query: {e}")
            self.outputs.append(
                (question, query_chain.generated_query, corrected_query, "", "")
            )
            return None

        final_output = query_chain.qa_chain.invoke(
            {"output": result, "input_question": question}
        ).strip("\n")

        logging.info(f"{final_output}")

        # add outputs of all steps to a list
        self.outputs.append(
            (
                question,
                query_chain.generated_query,
                corrected_query,
                result,
                final_output,
            )
        )

        return final_output

    def _make_query_chain_factory(self):
        """Return a zero-arg callable that creates a fresh QueryChain."""
        def factory():
            if isinstance(self.llm, dict):
                return QueryChain(
                    cypher_llm=self.llm["cypher_llm"],
                    qa_llm=self.llm["qa_llm"],
                    schema=self.neo4j_connection.schema,
                    search_type=self.search_type,
                    use_entity_centric_resolver=self.use_entity_centric_resolver,
                    resolver_config=self.resolver_config,
                    neo4j_helper=self.neo4j_connection.graph_helper,
                )
            return QueryChain(
                cypher_llm=self.llm,
                qa_llm=self.llm,
                schema=self.neo4j_connection.schema,
                search_type=self.search_type,
                use_entity_centric_resolver=self.use_entity_centric_resolver,
                resolver_config=self.resolver_config,
                neo4j_helper=self.neo4j_connection.graph_helper,
            )
        return factory

    def run_multi_hop(
        self,
        question: str,
        max_steps: int = 5,
        reset_llm_type: bool = False,
        model_name: Union[
            str, list[str], dict[Literal["cypher_llm_model", "qa_llm_model"], str]
        ] = None,
    ) -> dict:
        """
        Run multi-hop reasoning over the KG.

        At each hop the LLM decides to:
          A. continue exploring the current node,
          B. jump to another node,
          C. stop and answer, or
          D. run a global overview query.

        Returns a dict with keys: evidence, trace, answer.
        """
        if reset_llm_type and model_name:
            self.define_llm(model_name=model_name)

        logging.info(f"Starting multi-hop reasoning for: {question}")

        decision_llm = self.llm["cypher_llm"] if isinstance(self.llm, dict) else self.llm
        qa_llm = self.llm["qa_llm"] if isinstance(self.llm, dict) else self.llm

        reasoner = MultiHopReasoner(
            llm=decision_llm,
            neo4j_connection=self.neo4j_connection,
            query_chain_factory=self._make_query_chain_factory(),
            max_steps=max_steps,
            search_type=self.search_type,
        )

        hop_result = reasoner.run(question, top_k=self.top_k)

        # Generate final natural-language answer from accumulated evidence
        evidence = hop_result.get("evidence", [])
        qa_chain = CYPHER_OUTPUT_PARSER_PROMPT | qa_llm | StrOutputParser()
        final_output = evidence if evidence else "Given cypher query did not return any result"
        answer = qa_chain.invoke({
            "output": final_output,
            "input_question": question,
        }).strip("\n")

        logging.info(f"Multi-hop reasoning completed. Steps: {len(hop_result['trace'])}")

        return {
            "evidence": evidence,
            "trace": hop_result["trace"],
            "answer": answer,
        }

    def create_dataframe_from_outputs(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self.outputs,
            columns=[
                "Question",
                "Generated Query",
                "Corrected Query",
                "Query Result",
                "Natural Language Answer",
            ],
        )

        return df.replace("", np.nan)

    def handle_embedding(
        self, embedding: Union[np.array, None], vector_index: str
    ) -> Union[list[float], None]:
        if embedding is None:
            return None
        else:
            if np.isnan(embedding).any():
                raise ValueError("NaN value found in provided embedding")

            if np.isinf(embedding).any():
                raise ValueError("Infinite value found in provided embedding")

            if embedding.dtype != np.float_:
                raise ValueError("Input embedding must be a float array")

            if len(embedding.shape) > 1:
                raise ValueError("Input embedding must be a 1D array")

            vector_index_to_array_shape = {
                "SelformerEmbeddings": 768,
                "Prott5Embeddings": 1024,
                "Esm2Embeddings": 1280,
                "Anc2vecEmbeddings": 200,
                "CadaEmbeddings": 160,
                "Doc2vecEmbeddings": 100,
                "Dom2vecEmbeddings": 50,
                "RxnfpEmbeddings": 256,
                "BiokeenEmbeddings": 200,
            }

            if embedding.shape[0] != vector_index_to_array_shape[vector_index]:
                raise ValueError(
                    f"Invalid embedding vector shape provided. Expected {vector_index_to_array_shape[vector_index]}, got {embedding.shape[0]}"
                )

            return embedding.tolist()


def main():
    """
    Main function to execute the flow of operations.
    It initializes all necessary classes and executes the query generation, execution, and parsing process.
    """

    current_date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_filename = f"query_log_{current_date}.log"

    verbose_input = input("Enable verbose mode? (yes/no):\n").lower() == "yes"
    configure_logging(verbose=verbose_input, log_filename=log_filename)

    logging.info("Starting the pipeline...")

    try:
        pipeline = RunPipeline(
            verbose=verbose_input, model_name="gpt-3.5-turbo-instruct"
        )

        question = str(input("Enter a question:\n"))

        final_output = pipeline.run(question=question)

        print(final_output)

        logging.info("Pipeline finished successfully.")

    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        raise e


if __name__ == "__main__":
    main()
