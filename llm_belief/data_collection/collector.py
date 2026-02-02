"""Unified LLM data collection class.

This module provides the PairwiseCollector class for running various
pairwise comparison experiments with LLMs.
"""

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import pandas as pd
from openai import OpenAI
import numpy as np

import sys
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import get_config
from llm_belief.utils.paths import get_data_path, get_output_path, get_logs_path
from llm_belief.utils.attributes import (
    random_label_only,
    rearrange_dataframe,
    format_profile_for_prompt,
    get_real_profiles,
)
from llm_belief.utils.logging_setup import get_experiment_logger
from llm_belief.preprocessing import resample_profile_ids
from .prompts import get_prompt_variant


class PairwiseCollector:
    """Collector for pairwise comparison experiments with LLMs.

    This class provides methods for running different types of pairwise
    comparison experiments:
        - basic: Compare pairs of makeup profiles
        - fixreal: Compare real iPhone profiles with makeup profiles
        - top: Compare real profiles with top-scored profiles

    Attributes:
        cfg: Configuration object
        client: OpenAI API client
        model: Model name to use
        temperature: Sampling temperature
        reasoning_effort: Reasoning effort level
    """

    def __init__(
        self,
        api_key_env_var: Optional[str] = None,
        api_key: Optional[str] = None,
        logprobs: Optional[str] = None,
    ):
        """Initialize the collector.

        Args:
            api_key_env_var: Environment variable name for API key.
                           If not provided, uses config default.
        """
        self.cfg = get_config()
        self.model = self.cfg.get('openai', 'model')
        self.temperature = self.cfg.get('openai', 'temperature')
        self.reasoning_effort = self.cfg.get('openai', 'reasoning_effort', default='medium')
        logprobs_cfg = self.cfg.get("openai", "logprobs", default={})
        default_logprobs_enabled = bool(logprobs_cfg.get("enabled", False))
        if logprobs is None:
            self.logprobs_enabled = default_logprobs_enabled
        else:
            self.logprobs_enabled = (logprobs == "on")
        self.logprobs_model = logprobs_cfg.get("model", self.model)
        self.logprobs_temperature = logprobs_cfg.get("temperature", 0.0)
        self.logprobs_max_output_tokens = logprobs_cfg.get("max_output_tokens", 16)
        self.logprobs_top_logprobs = logprobs_cfg.get("top_logprobs", 2)
        self.logprobs_include = logprobs_cfg.get(
            "include",
            ["message.output_text.logprobs"],
        )

        # Get API key
        if api_key is None:
            api_key = self.cfg.get_api_key(api_key_env_var)
        self.client = OpenAI(api_key=api_key)

        # Set random seed
        random.seed(self.cfg.get('project', 'random_seed', default=2025))

    def _get_output_columns(self) -> List[str]:
        """Get standard output CSV columns."""
        cols = [
            "model",
            "temperature",
            "pair_id",
            "pair",
            "prompt_variant",
            "prompt",
            "prompt_response",
            "chosen_profile",
            "profile_id",
        ]
        if self.logprobs_enabled:
            cols += ["prob_chosen", "prob_nochosen"]
        return cols

    def _log_run_config(self, logger: logging.Logger) -> None:
        """Log the actual model settings used for the API call."""
        if self.logprobs_enabled:
            logger.info(
                "Using logprobs model: %s with temperature: %s, max_output_tokens: %s, top_logprobs: %s",
                self.logprobs_model,
                self.logprobs_temperature,
                self.logprobs_max_output_tokens,
                self.logprobs_top_logprobs,
            )
        else:
            logger.info(
                "Using model: %s with temperature: %s, reasoning_effort: %s",
                self.model,
                self.temperature,
                self.reasoning_effort,
            )

    @staticmethod
    def _get_field(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _extract_logprobs(
        self,
        response: Any,
        chosen_text: str,
    ) -> tuple[Optional[float], Optional[float]]:
        output = self._get_field(response, "output", []) or []
        if not output:
            return None, None
        content = self._get_field(output[0], "content", []) or []
        if not content:
            return None, None
        logprobs = self._get_field(content[0], "logprobs", None)
        if not logprobs:
            return None, None

        token_info = logprobs[0]
        token = self._get_field(token_info, "token", "")
        logprob = self._get_field(token_info, "logprob", None)
        top_logprobs = self._get_field(token_info, "top_logprobs", []) or []

        chosen_norm = (chosen_text or "").strip()
        token_norm = (token or "").strip()
        prob_chosen = None
        prob_nochosen = None

        if logprob is not None and token_norm == chosen_norm:
            prob_chosen = float(np.exp(logprob))

        if top_logprobs:
            for item in top_logprobs:
                t = self._get_field(item, "token", "")
                lp = self._get_field(item, "logprob", None)
                if lp is None:
                    continue
                if (t or "").strip() == chosen_norm:
                    prob_chosen = float(np.exp(lp))
                elif prob_nochosen is None:
                    prob_nochosen = float(np.exp(lp))

        if prob_nochosen is None and top_logprobs:
            for item in top_logprobs:
                t = self._get_field(item, "token", "")
                lp = self._get_field(item, "logprob", None)
                if lp is None:
                    continue
                if (t or "").strip() != chosen_norm:
                    prob_nochosen = float(np.exp(lp))
                    break

        return prob_chosen, prob_nochosen

    def _call_api(
        self,
        prompt: List[Dict[str, str]],
        reasoning_effort: Optional[str] = None,
    ) -> tuple[str, Optional[float], Optional[float]]:
        """Call OpenAI API with the given prompt.

        Args:
            prompt: List of message dictionaries

        Returns:
            Model response text
        """
        effort = reasoning_effort or self.reasoning_effort
        if self.logprobs_enabled:
            response = self.client.responses.create(
                model=self.logprobs_model,
                input=prompt,
                temperature=self.logprobs_temperature,
                max_output_tokens=self.logprobs_max_output_tokens,
                top_logprobs=self.logprobs_top_logprobs,
                include=self.logprobs_include,
            )
            text = response.output_text
            prob_chosen, prob_nochosen = self._extract_logprobs(response, text)
            return text, prob_chosen, prob_nochosen

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            reasoning={"effort": effort},
        )
        return response.output_text, None, None

    def _get_prompt_pair(
        self,
        labels: List[str],
        profiles: Dict[str, Dict[str, Any]],
        pair_id: int,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Return prompt pair/labels with alternating order."""
        prompt_labels = labels if pair_id % 2 == 0 else [labels[1], labels[0]]
        prompt_pair = [profiles[prompt_labels[0]], profiles[prompt_labels[1]]]
        return prompt_pair, prompt_labels

    def collect_basic(
        self,
        start_idx: int,
        end_idx: int,
        reasoning_effort: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Path:
        """Run basic pairwise comparison experiment.

        Compares pairs of makeup profiles from the shuffled profiles dataset.

        Args:
            start_idx: Starting pair index
            end_idx: Ending pair index (exclusive)
            reasoning_effort: Optional reasoning effort override
            output_file: Optional output filename. Defaults to '{start}_{end}.csv'

        Returns:
            Path to output CSV file
        """
        # Setup logging
        logger = get_experiment_logger("pair", f"{start_idx}_{end_idx}")
        self._log_run_config(logger)

        # Load profiles
        profiles_file = self.cfg.get('collection', 'profiles_file')
        df = pd.read_csv(get_data_path(profiles_file))
        df = rearrange_dataframe(df)

        if end_idx > len(df) // 2:
            raise ValueError(f"end_idx ({end_idx}) exceeds number of profile pairs ({len(df) // 2})")

        # Setup output
        if output_file is None:
            output_file = f"{start_idx}_{end_idx}.csv"
        csv_path = get_output_path(output_file)

        cols = self._get_output_columns()
        file_exists = csv_path.is_file()

        # Main loop
        start_time = datetime.now()
        last_call_time = start_time
        logger.info(f"API session start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()

            for pair_id in range(start_idx, end_idx):
                labels = random_label_only()
                profiles = {
                    labels[0]: df.iloc[2 * pair_id, :].to_dict(),
                    labels[1]: df.iloc[2 * pair_id + 1, :].to_dict(),
                }

                prompt_variant = pair_id % 10
                prompt_pair, prompt_labels = self._get_prompt_pair(
                    labels,
                    profiles,
                    pair_id,
                )
                prompt = get_prompt_variant(
                    prompt_variant,
                    prompt_pair,
                    prompt_labels,
                )

                res, prob_chosen, prob_nochosen = self._call_api(
                    prompt,
                    reasoning_effort=reasoning_effort,
                )

                call_time = datetime.now()
                logger.info(
                    f"Pair {pair_id}: {round((call_time - last_call_time).total_seconds())}s"
                )
                last_call_time = call_time

                chosen_profile = profiles.get(res)
                profile_id = labels.index(res) + 2 * pair_id if chosen_profile else None

                row = {
                    "model": self.logprobs_model if self.logprobs_enabled else self.model,
                    "temperature": self.logprobs_temperature if self.logprobs_enabled else self.temperature,
                    "pair_id": pair_id,
                    "pair": profiles,
                    "prompt_variant": prompt_variant,
                    "prompt": prompt,
                    "prompt_response": res,
                    "chosen_profile": chosen_profile,
                    "profile_id": profile_id,
                }
                if self.logprobs_enabled:
                    row["prob_chosen"] = prob_chosen
                    row["prob_nochosen"] = prob_nochosen
                writer.writerow(row)

        end_time = datetime.now()
        duration = round((end_time - start_time).total_seconds())
        logger.info(f"API session end: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}s")
        logger.info("All tasks completed successfully.")

        return csv_path

    def collect_fixreal(
        self,
        real_profile_id: str,
        n_makeup: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Path:
        """Run real vs. makeup profile comparison experiment.

        Samples makeup profiles from the first fixreal_sample_limit scored rows
        and persists their profile_id values under data/sample{n_makeup}_profile_ids.npy.

        Args:
            real_profile_id: ID of the real iPhone profile (e.g., 'iPhone 16 Pro')
            n_makeup: Number of makeup profiles to compare. Defaults to config value.
            reasoning_effort: Optional reasoning effort override
            output_file: Optional output filename

        Returns:
            Path to output CSV file
        """
        if n_makeup is None:
            n_makeup = self.cfg.get('collection', 'default_n_makeup', default=5000)
        sample_limit = self.cfg.get('collection', 'fixreal_sample_limit', default=20000)
        seed = self.cfg.get('project', 'random_seed', default=2025)

        # Setup logging
        safe_id = real_profile_id.replace(" ", "_")
        logger = get_experiment_logger("fixreal", safe_id)
        self._log_run_config(logger)
        logger.info(f"Real profile: {real_profile_id}, n_makeup: {n_makeup}")

        # Load real profiles from config
        real_profiles = get_real_profiles()
        if real_profile_id not in real_profiles:
            raise ValueError(f"Real profile '{real_profile_id}' not found")
        real_profile = format_profile_for_prompt(real_profiles[real_profile_id])

        # Load generated profiles for comparison
        profiles_file = self.cfg.get('collection', 'profiles_file')
        scored_df = pd.read_csv(get_data_path(profiles_file))
        sample_ids_file = f"sample{n_makeup}_profile_ids.npy"
        sample_ids = resample_profile_ids(
            scored_df,
            n_makeup=n_makeup,
            sample_limit=sample_limit,
            seed=seed,
            output_file=sample_ids_file,
            use_existing=True,
        )

        scoped_df = scored_df.iloc[: min(sample_limit, len(scored_df))]
        try:
            makeup_df = scoped_df.set_index("profile_id").loc[sample_ids].reset_index()
        except KeyError as exc:
            raise ValueError(
                "Sampled profile ids not found in scored profiles."
            ) from exc

        base_cols = list(scored_df.columns[:10]) + ["profile_id"]
        base_cols = [c for c in base_cols if c in makeup_df.columns]
        makeup_df = rearrange_dataframe(makeup_df[base_cols])
        makeup_profiles = makeup_df.to_dict(orient="records")

        # Setup output
        if output_file is None:
            output_file = f"{safe_id}_fixreal{n_makeup}.csv"
        csv_path = get_output_path(output_file)

        cols = self._get_output_columns()
        file_exists = csv_path.is_file()

        # Main loop
        start_time = datetime.now()
        last_call_time = start_time
        logger.info(f"API session start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()

            for pair_id, makeup_profile in enumerate(makeup_profiles):
                makeup_profile_id = makeup_profile.get("profile_id")
                makeup_prompt = {
                    k: v for k, v in makeup_profile.items() if k != "profile_id"
                }
                labels = random_label_only()
                profiles = {
                    labels[0]: real_profile,
                    labels[1]: makeup_prompt,
                }

                prompt_variant = pair_id % 10
                prompt_pair, prompt_labels = self._get_prompt_pair(
                    labels,
                    profiles,
                    pair_id,
                )
                prompt = get_prompt_variant(
                    prompt_variant,
                    prompt_pair,
                    prompt_labels,
                )

                res, prob_chosen, prob_nochosen = self._call_api(
                    prompt,
                    reasoning_effort=reasoning_effort,
                )

                call_time = datetime.now()
                logger.info(
                    f"Pair {pair_id}: {round((call_time - last_call_time).total_seconds())}s"
                )
                last_call_time = call_time

                chosen_profile = profiles.get(res)
                is_real_chosen = (res == labels[0])

                row = {
                    "model": self.logprobs_model if self.logprobs_enabled else self.model,
                    "temperature": self.logprobs_temperature if self.logprobs_enabled else self.temperature,
                    "pair_id": pair_id,
                    "pair": profiles,
                    "prompt_variant": prompt_variant,
                    "prompt": prompt,
                    "prompt_response": res,
                    "chosen_profile": chosen_profile,
                    "profile_id": real_profile_id if is_real_chosen else makeup_profile_id,
                }
                if self.logprobs_enabled:
                    row["prob_chosen"] = prob_chosen
                    row["prob_nochosen"] = prob_nochosen
                writer.writerow(row)

        end_time = datetime.now()
        duration = round((end_time - start_time).total_seconds())
        logger.info(f"API session end: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}s")

        return csv_path

    def collect_top(
        self,
        real_profile_id: str,
        n_top: Optional[int] = None,
        score_column: str = "MLP_score",
        reasoning_effort: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Path:
        """Run real vs. top-scored profile comparison experiment.

        Args:
            real_profile_id: ID of the real iPhone profile
            n_top: Number of top profiles to compare. Defaults to config value.
            score_column: Column name for sorting profiles
            reasoning_effort: Optional reasoning effort override
            output_file: Optional output filename

        Returns:
            Path to output CSV file
        """
        if n_top is None:
            n_top = self.cfg.get('collection', 'default_n_top', default=50)

        # Setup logging
        safe_id = real_profile_id.replace(" ", "_")
        logger = get_experiment_logger("top", f"{safe_id}_ntop{n_top}")
        self._log_run_config(logger)
        logger.info(f"Real profile: {real_profile_id}, n_top: {n_top}")

        # Load real profiles from config
        real_profiles = get_real_profiles()
        if real_profile_id not in real_profiles:
            raise ValueError(f"Real profile '{real_profile_id}' not found")
        real_profile = format_profile_for_prompt(real_profiles[real_profile_id])

        # Load and sort scored profiles
        scored_file = self.cfg.get('collection', 'scored_profiles_file')
        scored_df = pd.read_csv(get_data_path(scored_file))

        if score_column not in scored_df.columns:
            raise ValueError(f"Score column '{score_column}' not found")

        scored_df = scored_df.sort_values(by=score_column, ascending=False)
        top_df = scored_df.head(n_top)
        top_df_display = rearrange_dataframe(top_df.iloc[:, :10])

        # Setup output
        if output_file is None:
            output_file = f"{safe_id}_ntop{n_top}.csv"
        csv_path = get_output_path(output_file)

        cols = self._get_output_columns()
        file_exists = csv_path.is_file()

        # Main loop
        start_time = datetime.now()
        last_call_time = start_time

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()

            for pair_id in range(len(top_df_display)):
                labels = random_label_only()
                top_profile = top_df_display.iloc[pair_id].to_dict()
                profiles = {
                    labels[0]: real_profile,
                    labels[1]: top_profile,
                }

                prompt_variant = pair_id % 10
                prompt_pair, prompt_labels = self._get_prompt_pair(
                    labels,
                    profiles,
                    pair_id,
                )
                prompt = get_prompt_variant(
                    prompt_variant,
                    prompt_pair,
                    prompt_labels,
                )

                res, prob_chosen, prob_nochosen = self._call_api(
                    prompt,
                    reasoning_effort=reasoning_effort,
                )

                call_time = datetime.now()
                logger.info(
                    f"Pair {pair_id}: {round((call_time - last_call_time).total_seconds())}s"
                )
                last_call_time = call_time

                chosen_profile = profiles.get(res)
                is_real_chosen = (res == labels[0])

                row = {
                    "model": self.logprobs_model if self.logprobs_enabled else self.model,
                    "temperature": self.logprobs_temperature if self.logprobs_enabled else self.temperature,
                    "pair_id": pair_id,
                    "pair": profiles,
                    "prompt_variant": prompt_variant,
                    "prompt": prompt,
                    "prompt_response": res,
                    "chosen_profile": chosen_profile,
                    "profile_id": real_profile_id if is_real_chosen else f"top_{pair_id}",
                }
                if self.logprobs_enabled:
                    row["prob_chosen"] = prob_chosen
                    row["prob_nochosen"] = prob_nochosen
                writer.writerow(row)

        end_time = datetime.now()
        duration = round((end_time - start_time).total_seconds())
        logger.info(f"Total duration: {duration}s")

        return csv_path

    def collect_context_fixreal(
        self,
        real_profile_id: str,
        context_file: str,
        sample_ids_file: str = "sample5k_profile_ids.npy",
        scored_limit: int = 20000,
        output_file: Optional[str] = None,
        context_date: str = "2025-03-15",
        reasoning_effort: Optional[str] = None,
    ) -> Path:
        """Run fixreal with injected context from a text file.

        Args:
            real_profile_id: ID of the real iPhone profile
            context_file: Path to context text file (relative to data/ or absolute)
            sample_ids_file: Numpy file with sampled profile ids
            scored_limit: Max rows from profiles to consider
            output_file: Optional output filename
            context_date: Date to include in system context
            reasoning_effort: Optional reasoning effort override
        """
        safe_id = real_profile_id.replace(" ", "_")
        logger = get_experiment_logger("reali16_fixreal", safe_id)
        self._log_run_config(logger)
        logger.info(f"Real profile: {real_profile_id}")

        profiles_file = self.cfg.get("collection", "profiles_file")
        profiles_df = pd.read_csv(get_data_path(profiles_file))

        sample_limit = self.cfg.get("collection", "fixreal_sample_limit", default=20000)
        if scored_limit is not None:
            sample_limit = scored_limit
        n_makeup = self.cfg.get("collection", "default_n_makeup", default=5000)
        seed = self.cfg.get("project", "random_seed", default=2025)

        sample_ids = resample_profile_ids(
            profiles_df,
            n_makeup=n_makeup,
            sample_limit=sample_limit,
            seed=seed,
            output_file=sample_ids_file,
            use_existing=True,
        )

        scoped_df = profiles_df.iloc[: min(sample_limit, len(profiles_df))]
        try:
            sampled = scoped_df.set_index("profile_id").loc[sample_ids].reset_index()
        except KeyError as exc:
            raise ValueError(
                "Sampled profile ids not found in profiles."
            ) from exc

        base_cols = list(profiles_df.columns)
        df = rearrange_dataframe(sampled[base_cols]).reset_index(drop=True)

        real_profiles = get_real_profiles()
        if real_profile_id not in real_profiles:
            raise ValueError(f"Real profile '{real_profile_id}' not found")
        real_profile_formatted = format_profile_for_prompt(real_profiles[real_profile_id])

        context_path = Path(context_file)
        if not context_path.is_absolute():
            context_path = get_data_path(context_file)
        if not context_path.is_file():
            raise FileNotFoundError(f"Context file not found: {context_path}")
        with open(context_path, "r", encoding="utf-8") as f:
            context_text = f.read()

        external_knowledge = [
            {
                "role": "system",
                "content": (
                    "The following context is provided:\n"
                    f"{context_text}\n"
                ),
            }
        ]

        # Setup output
        if output_file is None:
            output_file = f"context_{safe_id}_fixreal_{len(df)}.csv"
        csv_path = get_output_path(output_file)
        cols = self._get_output_columns()
        file_exists = csv_path.is_file()

        start_time = datetime.now()
        last_call_time = start_time
        logger.info(f"API session start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()

            for pair_id in range(len(df)):
                labels = random_label_only()
                profile_id = df.loc[pair_id, "profile_id"]
                if pair_id % 2 == 0:
                    profiles = {
                        labels[0]: real_profile_formatted,
                        labels[1]: df.iloc[pair_id, :10].to_dict(),
                    }
                    ids = [real_profile_id, profile_id]
                else:
                    profiles = {
                        labels[0]: df.iloc[pair_id, :10].to_dict(),
                        labels[1]: real_profile_formatted,
                    }
                    ids = [profile_id, real_profile_id]

                prompt_variant = pair_id % 10
                prompt = get_prompt_variant(
                    prompt_variant,
                    list(profiles.values()),
                    labels,
                    date_override=context_date,
                )
                prompt = external_knowledge + prompt
                res, prob_chosen, prob_nochosen = self._call_api(
                    prompt,
                    reasoning_effort=reasoning_effort,
                )

                call_time = datetime.now()
                logger.info(
                    f"Pair {pair_id}: {round((call_time - last_call_time).total_seconds())}s"
                )
                last_call_time = call_time

                chosen_profile = profiles.get(res)
                chosen_id = ids[labels.index(res)] if chosen_profile else None

                row = {
                    "model": self.logprobs_model if self.logprobs_enabled else self.model,
                    "temperature": self.logprobs_temperature if self.logprobs_enabled else self.temperature,
                    "pair_id": pair_id,
                    "pair": profiles,
                    "prompt_variant": prompt_variant,
                    "prompt": prompt,
                    "prompt_response": res,
                    "chosen_profile": chosen_profile,
                    "profile_id": chosen_id,
                }
                if self.logprobs_enabled:
                    row["prob_chosen"] = prob_chosen
                    row["prob_nochosen"] = prob_nochosen
                writer.writerow(row)

        end_time = datetime.now()
        duration = round((end_time - start_time).total_seconds())
        logger.info(f"API session end: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}s")

        return csv_path

    def collect_rag_fixreal(
        self,
        real_profile_id: str,
        n_makeup: Optional[int] = None,
        exclude_ids_file: Optional[str] = "fixreal_used_profile_ids.npy",
        rag_faiss: Optional[str] = None,
        rag_meta: Optional[str] = None,
        rag_k: int = 3,
        rag_per_chars: int = 1200,
        rag_embed_model: str = "text-embedding-3-small",
        output_file: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Path:
        """Run fixreal with RAG context prepended."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "RAG dependencies missing. Install optional 'rag' dependencies."
            ) from exc

        if n_makeup is None:
            n_makeup = self.cfg.get("collection", "default_n_makeup", default=5000)
        sample_limit = self.cfg.get("collection", "fixreal_sample_limit", default=20000)
        seed = self.cfg.get("project", "random_seed", default=2025)

        safe_id = real_profile_id.replace(" ", "_")
        logger = get_experiment_logger("rag_fixreal", safe_id)
        self._log_run_config(logger)
        logger.info(f"Real profile: {real_profile_id}, n_makeup: {n_makeup}")

        rag_faiss = rag_faiss or os.getenv("RAG_FAISS")
        rag_meta = rag_meta or os.getenv("RAG_META")
        if not rag_faiss or not rag_meta:
            raise ValueError("RAG_FAISS and RAG_META must be provided for RAG runs.")

        # Load real profiles from config
        real_profiles = get_real_profiles()
        if real_profile_id not in real_profiles:
            raise ValueError(f"Real profile '{real_profile_id}' not found")
        real_profile = format_profile_for_prompt(real_profiles[real_profile_id])

        # Load generated profiles and reuse fixed sample ids
        profiles_file = self.cfg.get("collection", "profiles_file")
        profiles_df = pd.read_csv(get_data_path(profiles_file))

        sample_ids = resample_profile_ids(
            profiles_df,
            n_makeup=n_makeup,
            sample_limit=sample_limit,
            seed=seed,
            output_file=f"sample{n_makeup}_profile_ids.npy",
            use_existing=True,
        )
        if exclude_ids_file:
            exclude_ids = np.load(get_data_path(exclude_ids_file), allow_pickle=True)
            sample_ids = np.array(
                [pid for pid in sample_ids if pid not in set(exclude_ids)]
            )
            if len(sample_ids) == 0:
                raise ValueError(
                    "All sampled profile ids were excluded; regenerate sample ids or "
                    "adjust exclude_ids_file."
                )

        scoped_df = profiles_df.iloc[: min(sample_limit, len(profiles_df))]
        try:
            makeup_df = scoped_df.set_index("profile_id").loc[sample_ids].reset_index()
        except KeyError as exc:
            raise ValueError(
                "Sampled profile ids not found in profiles."
            ) from exc

        base_cols = list(profiles_df.columns)
        makeup_df = rearrange_dataframe(makeup_df[base_cols])
        makeup_profiles = makeup_df.to_dict(orient="records")

        # RAG helpers
        def _embed_texts(texts: List[str], batch: int = 64) -> List[np.ndarray]:
            out = []
            for i in range(0, len(texts), batch):
                part = texts[i : i + batch]
                resp = self.client.embeddings.create(model=rag_embed_model, input=part)
                for d in resp.data:
                    out.append(np.array(d.embedding, dtype="float32"))
            return out

        def _to_query_str(q: Any) -> str:
            if isinstance(q, str):
                return q
            if isinstance(q, dict):
                return q.get("content", "")
            if isinstance(q, list):
                return " ".join(_to_query_str(e) for e in q)
            return str(q)

        def _truncate_chars(text: str, max_chars: int) -> str:
            return text if len(text) <= max_chars else text[:max_chars]

        def _load_index_and_meta(faiss_path: str, meta_path: str):
            index = faiss.read_index(faiss_path)
            meta = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        meta.append(json.loads(line))
            return index, meta

        def _search(
            query: str, index, meta_list
        ) -> List[tuple[Dict[str, Any], float]]:
            query_str = _to_query_str(query).strip()
            if not query_str:
                raise ValueError("Empty query for retrieval.")
            qv = _embed_texts([query_str], batch=1)[0]
            qv = qv / (np.linalg.norm(qv) + 1e-12)
            D, I = index.search(np.expand_dims(qv, 0), rag_k)
            hits = []
            for j, idx in enumerate(I[0]):
                if 0 <= idx < len(meta_list):
                    hits.append((meta_list[idx], float(D[0][j])))
            return hits

        def _build_context(
            hits: List[tuple[Dict[str, Any], float]],
        ) -> tuple[str, List[Dict[str, Any]]]:
            blocks = []
            sources = []
            for i, (d, score) in enumerate(hits, start=1):
                text = _truncate_chars(d.get("text", ""), rag_per_chars)
                src = d.get("source_url") or d.get("source_path") or ""
                title = d.get("title") or os.path.basename(src) or "(untitled)"
                blocks.append(f"[score={score:.3f}] {title}\n{src}\n{text}")
                sources.append({"id": f"S{i}", "title": title, "source": src, "score": score})
            return "\n\n---\n\n".join(blocks), sources

        def _prepend_rag_to_prompt(
            original_prompt: List[Dict[str, Any]],
        ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            hits = _search(original_prompt, rag_index, rag_meta_list)
            ctx, sources = _build_context(hits)
            rag_header = (
                "Retrieved External Context (for reference)\n"
                f"{ctx}\n\n"
                "You may combine the external context above with your own internal knowledge "
                "to make the most likely judgement.\n"
            )
            return [{"role": "system", "content": rag_header}] + original_prompt, sources

        rag_index, rag_meta_list = _load_index_and_meta(rag_faiss, rag_meta)

        # Setup output
        if output_file is None:
            output_file = f"RAG_{safe_id}_fixreal_{len(makeup_profiles)}.csv"
        csv_path = get_output_path(output_file)
        cols = self._get_output_columns() + ["retrieval_context", "retrieval_hits"]
        file_exists = csv_path.is_file()

        start_time = datetime.now()
        last_call_time = start_time
        logger.info(f"API session start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()

            for pair_id, makeup_profile in enumerate(makeup_profiles):
                labels = random_label_only()
                profiles = {
                    labels[0]: real_profile,
                    labels[1]: makeup_profile,
                }

                prompt_variant = pair_id % 10
                prompt_pair, prompt_labels = self._get_prompt_pair(
                    labels,
                    profiles,
                    pair_id,
                )
                base_prompt = get_prompt_variant(
                    prompt_variant,
                    prompt_pair,
                    prompt_labels,
                )
                final_prompt, sources = _prepend_rag_to_prompt(base_prompt)
                res, prob_chosen, prob_nochosen = self._call_api(
                    final_prompt,
                    reasoning_effort=reasoning_effort,
                )

                call_time = datetime.now()
                logger.info(
                    f"Pair {pair_id}: {round((call_time - last_call_time).total_seconds())}s"
                )
                last_call_time = call_time

                chosen_profile = profiles.get(res)
                is_real_chosen = (res == labels[0])
                makeup_profile_id = makeup_profile.get("profile_id")

                row = {
                    "model": self.logprobs_model if self.logprobs_enabled else self.model,
                    "temperature": self.logprobs_temperature if self.logprobs_enabled else self.temperature,
                    "pair_id": pair_id,
                    "pair": profiles,
                    "prompt_variant": prompt_variant,
                    "prompt": final_prompt,
                    "prompt_response": res,
                    "chosen_profile": chosen_profile,
                    "profile_id": real_profile_id if is_real_chosen else makeup_profile_id,
                    "retrieval_context": f"{{'k': {rag_k}, 'per_chars': {rag_per_chars}}}",
                    "retrieval_hits": sources,
                }
                if self.logprobs_enabled:
                    row["prob_chosen"] = prob_chosen
                    row["prob_nochosen"] = prob_nochosen
                writer.writerow(row)

        end_time = datetime.now()
        duration = round((end_time - start_time).total_seconds())
        logger.info(f"API session end: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}s")

        return csv_path
