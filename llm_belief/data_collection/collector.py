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
from typing import Optional, List, Dict, Any, Tuple
import logging

import pandas as pd
from openai import OpenAI, PermissionDeniedError, AuthenticationError
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
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """Initialize the collector.

        Args:
            api_key_env_var: Environment variable name for API key.
                           If not provided, uses config default.
        """
        self.cfg = get_config()
        self.model = model or self.cfg.get('openai', 'model')
        self.temperature = (
            temperature
            if temperature is not None
            else self.cfg.get('openai', 'temperature')
        )
        self.reasoning_effort = self.cfg.get('openai', 'reasoning_effort', default='medium')
        self.reasoning_model_prefixes = tuple(
            str(prefix).lower()
            for prefix in self.cfg.get(
                "openai",
                "reasoning_model_prefixes",
                default=["gpt-5", "o"],
            )
        )
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
        attempted_model = self.logprobs_model if self.logprobs_enabled else self.model
        try:
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

            kwargs = {
                "model": self.model,
                "input": prompt,
                "temperature": self.temperature,
            }
            model_name = str(self.model or "").lower()
            if model_name.startswith(self.reasoning_model_prefixes):
                kwargs["reasoning"] = {"effort": effort}
            response = self.client.responses.create(**kwargs)
            return response.output_text, None, None
        except PermissionDeniedError as err:
            # Avoid dumping prompt contents; provide actionable config hints instead.
            body = getattr(err, "body", None)
            detail = None
            if isinstance(body, dict):
                detail = (body.get("error") or {}).get("message")
            raise RuntimeError(
                "OpenAI API returned 403 PermissionDenied.\n"
                f"  attempted_model: {attempted_model}\n"
                f"  logprobs_enabled: {self.logprobs_enabled}\n"
                "Common causes:\n"
                "- Your API key / project does not have access to this model.\n"
                "- The model is disabled by org/project policy.\n"
                "Fix:\n"
                "- If running with --logprobs on, change config/config.yaml → openai.logprobs.model.\n"
                "- Otherwise change config/config.yaml → openai.model.\n"
                + (f"\nProvider message: {detail}" if detail else "")
            ) from err
        except AuthenticationError as err:
            raise RuntimeError(
                "OpenAI API authentication failed.\n"
                "Check that your API key environment variable is set and points to the correct key.\n"
                f"  config openai.api_key_env_var: {self.cfg.get('openai', 'api_key_env_var')}"
            ) from err

    def _extract_label_probabilities(
        self,
        response: Any,
        labels: tuple[str, str],
    ) -> tuple[Optional[float], Optional[float]]:
        """Extract probabilities for two explicit labels from first-token logprobs."""
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
        token = (self._get_field(token_info, "token", "") or "").strip().upper()
        logprob = self._get_field(token_info, "logprob", None)
        top_logprobs = self._get_field(token_info, "top_logprobs", []) or []

        a, b = labels[0].upper(), labels[1].upper()
        prob_a: Optional[float] = None
        prob_b: Optional[float] = None

        if logprob is not None:
            if token == a:
                prob_a = float(np.exp(logprob))
            elif token == b:
                prob_b = float(np.exp(logprob))

        for item in top_logprobs:
            t = (self._get_field(item, "token", "") or "").strip().upper()
            lp = self._get_field(item, "logprob", None)
            if lp is None:
                continue
            if t == a:
                prob_a = float(np.exp(lp))
            elif t == b:
                prob_b = float(np.exp(lp))

        return prob_a, prob_b

    def _call_api_yesno(
        self,
        prompt: List[Dict[str, str]],
        reasoning_effort: Optional[str] = None,
    ) -> tuple[str, Optional[float], Optional[float]]:
        """Call API for Y/N task; optionally return prob_yes/prob_no when logprobs enabled."""
        effort = reasoning_effort or self.reasoning_effort
        attempted_model = self.logprobs_model if self.logprobs_enabled else self.model
        try:
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
                prob_yes, prob_no = self._extract_label_probabilities(response, ("Y", "N"))
                return text, prob_yes, prob_no

            kwargs = {
                "model": self.model,
                "input": prompt,
                "temperature": self.temperature,
            }
            model_name = str(self.model or "").lower()
            if model_name.startswith(self.reasoning_model_prefixes):
                kwargs["reasoning"] = {"effort": effort}
            response = self.client.responses.create(**kwargs)
            return response.output_text, None, None
        except PermissionDeniedError as err:
            body = getattr(err, "body", None)
            detail = None
            if isinstance(body, dict):
                detail = (body.get("error") or {}).get("message")
            raise RuntimeError(
                "OpenAI API returned 403 PermissionDenied.\n"
                f"  attempted_model: {attempted_model}\n"
                f"  logprobs_enabled: {self.logprobs_enabled}\n"
                "Common causes:\n"
                "- Your API key / project does not have access to this model.\n"
                "- The model is disabled by org/project policy.\n"
                "Fix:\n"
                "- If running with --logprobs on, change config/config.yaml → openai.logprobs.model.\n"
                "- Otherwise change config/config.yaml → openai.model.\n"
                + (f"\nProvider message: {detail}" if detail else "")
            ) from err
        except AuthenticationError as err:
            raise RuntimeError(
                "OpenAI API authentication failed.\n"
                "Check that your API key environment variable is set and points to the correct key.\n"
                f"  config openai.api_key_env_var: {self.cfg.get('openai', 'api_key_env_var')}"
            ) from err

    @staticmethod
    def _normalize_yes_no(response_text: str) -> str:
        s = (response_text or "").strip().upper()
        if not s:
            return ""
        first = s[0]
        return first if first in {"Y", "N"} else ""

    def _build_question_prompt(
        self,
        question: str,
        product: str,
        prompt_variant: int,
        date_override: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt messages for question-level Y/N elicitation."""
        date_text = date_override or self.cfg.get("prompting", "default_date", default="2024-06-01")
        idx = int(prompt_variant) % 10
        system_texts = [
            "You will be provided with a forecast of {product}'s specification, anticipated to launch within 6 months.",
            "You will review a projected specification statement for {product}, expected to launch within the next 6 months.",
            "A forecasted specification statement for {product} will be shown; treat it as a model expected within 6 months.",
            "You will evaluate a candidate specification forecast for {product}, anticipated over a 6-month horizon.",
            "Consider a potential specification statement for {product}, expected to debut in roughly 6 months.",
            "Assess a proposed specification forecast for {product}, which is anticipated to launch within approximately 6 months.",
            "You will judge a forecasted product statement for {product}, projected for release within the next 6 months.",
            "Evaluate whether a given specification statement is plausible for {product}, expected in about 6 months.",
            "You are given a forward-looking specification claim for {product}, expected to launch within a 6-month timeframe.",
            "Review a forecast statement about {product}'s specs, with launch anticipated within the next 6 months.",
        ]
        criteria_texts = [
            "Base your assessment on technical feasibility, product segmentation, performance/thermal constraints, and historical {product} generation trends.",
            "Base your assessment on engineering feasibility, product-tier positioning, realistic constraints, and historical {product} generation patterns.",
            "Base your assessment on what is technically plausible, how tiers are typically segmented, realistic constraints, and prior-generation trends for {product}.",
            "Base your assessment on architecture/manufacturing feasibility, expected segmentation, and historical roadmap trends for {product}.",
            "Base your assessment on feasibility and constraints, expected portfolio positioning, and patterns observed across past {product} generations.",
            "Base your assessment on practical feasibility, brand-style segmentation, plausible capability envelopes, and cross-generation trends for {product}.",
            "Base your assessment on what is realistic under constraints, consistent with product segmentation, and aligned with historical {product} trajectories.",
            "Base your assessment on technical plausibility, segmentation logic, and lessons from historical {product} generation trends.",
            "Base your assessment on feasibility considerations, product positioning, segmentation logic, and long-run trends across {product} generations.",
            "Base your assessment on technical feasibility, realistic capability envelopes, expected segmentation, and historical trendlines for {product}.",
        ]

        system_content = (
            f"Assume the current date is {date_text}. "
            + system_texts[idx].format(product=product)
        )
        user_content = (
            str(question).strip()
            + " Return exactly one label: 'Y' or 'N'. 'Y' for Yes and 'N' for No."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def collect_questions_csv(
        self,
        question_csv: str,
        product: str,
        question_column: str = "question",
        reasoning_effort: Optional[str] = None,
        output_file: Optional[str] = None,
        date_override: Optional[str] = None,
    ) -> Path:
        """Read CSV questions and ask Y/N for each row (single or 10 prompt variants)."""
        csv_path_in = Path(question_csv)
        if not csv_path_in.is_absolute():
            cwd_path = Path.cwd() / csv_path_in
            root_path = _project_root / csv_path_in
            if cwd_path.is_file():
                csv_path_in = cwd_path
            elif root_path.is_file():
                csv_path_in = root_path
        if not csv_path_in.is_file():
            raise FileNotFoundError(f"Question CSV not found: {question_csv}")

        qdf = pd.read_csv(csv_path_in)
        if qdf.empty:
            raise ValueError(f"Question CSV has no rows: {csv_path_in}")

        q_col = question_column
        if q_col not in qdf.columns:
            if len(qdf.columns) == 1:
                q_col = str(qdf.columns[0])
            else:
                raise ValueError(
                    f"Question column '{question_column}' not found in CSV columns: {list(qdf.columns)}"
                )

        logger = get_experiment_logger("questioncsv", Path(csv_path_in).stem)
        self._log_run_config(logger)
        logger.info("question_csv=%s, product=%s, question_column=%s", csv_path_in, product, q_col)

        effective_temp = self.logprobs_temperature if self.logprobs_enabled else self.temperature
        num_variants_cfg = int(self.cfg.get("collection", "num_prompt_variants", default=10) or 10)
        num_variants = 1 if float(effective_temp) == 0 else max(1, num_variants_cfg)

        if output_file is None:
            output_file = f"{Path(csv_path_in).stem}_questioncsv.csv"
            csv_path_out = get_output_path(output_file)
        else:
            out_arg = Path(output_file)
            if out_arg.is_absolute():
                csv_path_out = out_arg
            else:
                # Treat relative paths as project-root relative when they include folders,
                # otherwise keep legacy behavior under output/.
                if ("/" in output_file) or ("\\" in output_file):
                    csv_path_out = (_project_root / out_arg).resolve()
                else:
                    csv_path_out = get_output_path(output_file)
            csv_path_out.parent.mkdir(parents=True, exist_ok=True)

        cols = [
            "model",
            "temperature",
            "question",
            "prompt_variant",
            "prompt",
            "prompt_response",
            "answer_yes",
            "answer_no",
        ]
        if self.logprobs_enabled:
            cols += ["prob_yes", "prob_no"]

        file_exists = csv_path_out.is_file()
        completed: set[tuple[str, int]] = set()
        if file_exists and csv_path_out.stat().st_size > 0:
            try:
                existing = pd.read_csv(csv_path_out, usecols=["question", "prompt_variant"])
                for _, r in existing.iterrows():
                    q_raw = r.get("question")
                    pv_raw = r.get("prompt_variant")
                    if pd.isna(q_raw) or pd.isna(pv_raw):
                        continue
                    q_key = str(q_raw).strip()
                    try:
                        pv_key = int(pv_raw)
                    except Exception:
                        continue
                    completed.add((q_key, pv_key))
                logger.info(
                    "Resuming question-csv: found %s completed (question, prompt_variant) rows in %s",
                    len(completed),
                    csv_path_out,
                )
            except Exception:
                completed = set()
                logger.warning("Unable to parse existing question-csv output for resume; processing from scratch.")

        start_time = datetime.now()
        last_call_time = start_time
        logger.info("API session start: %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))

        with open(csv_path_out, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()

            row_id = 0
            for _, row in qdf.iterrows():
                question = row.get(q_col)
                if pd.isna(question):
                    continue
                question_text = str(question).strip()
                if not question_text:
                    continue

                for pv in range(num_variants):
                    if (question_text, pv) in completed:
                        continue
                    prompt = self._build_question_prompt(
                        question=question_text,
                        product=product,
                        prompt_variant=pv,
                        date_override=date_override,
                    )
                    res, prob_yes, prob_no = self._call_api_yesno(
                        prompt,
                        reasoning_effort=reasoning_effort,
                    )
                    normalized = self._normalize_yes_no(res)

                    call_time = datetime.now()
                    logger.info(
                        "Question row %s variant %s: %ss",
                        row_id,
                        pv,
                        round((call_time - last_call_time).total_seconds()),
                    )
                    last_call_time = call_time

                    out_row = {
                        "model": self.logprobs_model if self.logprobs_enabled else self.model,
                        "temperature": self.logprobs_temperature if self.logprobs_enabled else self.temperature,
                        "question": question_text,
                        "prompt_variant": pv,
                        "prompt": json.dumps(prompt, ensure_ascii=False),
                        "prompt_response": res,
                        "answer_yes": 1 if normalized == "Y" else 0,
                        "answer_no": 1 if normalized == "N" else 0,
                    }
                    if self.logprobs_enabled:
                        out_row["prob_yes"] = prob_yes
                        out_row["prob_no"] = prob_no
                    writer.writerow(out_row)
                    completed.add((question_text, pv))
                row_id += 1

        end_time = datetime.now()
        logger.info("API session end: %s", end_time.strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Total duration: %ss", round((end_time - start_time).total_seconds()))
        return csv_path_out

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

    def _get_real_profile_formatted(
        self,
        real_profile_id: str,
        real_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve and format a real profile for prompts.

        Args:
            real_profile_id: Identifier used for logging/output and (if needed) config lookup.
            real_profile: Optional explicit profile dict (bypasses config lookup).

        Returns:
            Formatted profile dict suitable for prompt injection.
        """
        def _format_like_makeup(profile_cfg_keys: Dict[str, Any]) -> Dict[str, Any]:
            attrs_cfg = self.cfg.get_attributes() or {}
            display_row: Dict[str, Any] = {}
            for key, value in (profile_cfg_keys or {}).items():
                cfg = attrs_cfg.get(key, {}) if isinstance(attrs_cfg, dict) else {}
                display_name = cfg.get("prompt_name") or cfg.get("name") or key
                display_row[display_name] = value

            df = pd.DataFrame([display_row])
            df = rearrange_dataframe(df)
            return df.iloc[0].to_dict()

        if real_profile is not None:
            if not isinstance(real_profile, dict):
                raise TypeError("real_profile must be a dict when provided")
            return _format_like_makeup(real_profile)

        real_profiles = get_real_profiles()
        if real_profile_id not in real_profiles:
            raise ValueError(f"Real profile '{real_profile_id}' not found")
        return _format_like_makeup(real_profiles[real_profile_id])

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
        real_profile: Optional[Dict[str, Any]] = None,
        alternative_profiles: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    ) -> Path:
        """Run real vs. makeup profile comparison experiment.

        Samples makeup profiles from the first fixreal_sample_limit scored rows
        and persists their profile_id values under data/sample{n_makeup}_profile_ids.npy.

        Args:
            real_profile_id: ID of the real iPhone profile (e.g., 'iPhone 16 Pro')
            n_makeup: Number of makeup profiles to compare. Defaults to config value.
            reasoning_effort: Optional reasoning effort override
            output_file: Optional output filename
            alternative_profiles: Optional fixed alternatives as a list of
                (profile_id, config-keyed-attributes) tuples. When provided,
                `n_makeup` sampling is skipped and alternatives are taken from
                this list directly.

        Returns:
            Path to output CSV file
        """
        # Setup logging
        safe_id = real_profile_id.replace(" ", "_")
        logger = get_experiment_logger("fixreal", safe_id)
        self._log_run_config(logger)

        # Load real profiles from config
        real_profile = self._get_real_profile_formatted(
            real_profile_id,
            real_profile=real_profile,
        )

        if alternative_profiles is not None:
            if n_makeup is not None:
                raise ValueError("n_makeup cannot be used when alternative_profiles is provided.")
            makeup_profiles: List[Dict[str, Any]] = []
            for alt_id, alt_profile in alternative_profiles:
                alt_formatted = self._get_real_profile_formatted(
                    str(alt_id),
                    real_profile=alt_profile,
                )
                alt_formatted["profile_id"] = str(alt_id)
                makeup_profiles.append(alt_formatted)
            logger.info(
                "Real profile: %s, fixed alternatives from CSV: %s",
                real_profile_id,
                len(makeup_profiles),
            )
        else:
            if n_makeup is None:
                n_makeup = self.cfg.get('collection', 'default_n_makeup', default=5000)
            sample_limit = self.cfg.get('collection', 'fixreal_sample_limit', default=20000)
            seed = self.cfg.get('project', 'random_seed', default=2025)
            logger.info(f"Real profile: {real_profile_id}, n_makeup: {n_makeup}")

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
            if alternative_profiles is not None:
                output_file = f"{safe_id}_fixreal_altset{len(makeup_profiles)}.csv"
            else:
                output_file = f"{safe_id}_fixreal{n_makeup}.csv"
        csv_path = get_output_path(output_file)

        cols = self._get_output_columns()
        file_exists = csv_path.is_file()

        existing_pair_ids: set[int] = set()
        if file_exists and csv_path.stat().st_size > 0:
            try:
                with open(csv_path, mode="r", newline="", encoding="utf-8") as rf:
                    reader = csv.DictReader(rf)
                    for row in reader:
                        raw = row.get("pair_id")
                        if raw is None:
                            continue
                        try:
                            pid = int(raw)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= pid < len(makeup_profiles):
                            existing_pair_ids.add(pid)
            except OSError:
                existing_pair_ids = set()
        if existing_pair_ids:
            logger.info(
                "Resuming %s: found %s existing rows; will skip those pair_id values.",
                csv_path.name,
                len(existing_pair_ids),
            )

        # Main loop
        start_time = datetime.now()
        last_call_time = start_time
        logger.info(f"API session start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if (not file_exists) or csv_path.stat().st_size == 0:
                writer.writeheader()

            for pair_id, makeup_profile in enumerate(makeup_profiles):
                if pair_id in existing_pair_ids:
                    continue

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

    def collect_allcomb(
        self,
        real_profiles: List[Tuple[str, Dict[str, Any]]],
        reasoning_effort: Optional[str] = None,
        output_file: Optional[str] = None,
        context_file: Optional[str] = None,
        context_date: Optional[str] = None,
    ) -> Path:
        """Run all-combinations (all-pairs) comparisons across a provided set of real profiles.

        This mode compares N provided real profiles in a round-robin fashion,
        producing N*(N-1)/2 pairwise judgments.

        Args:
            real_profiles: List of (real_profile_id, profile_dict) where profile_dict
                is keyed by attribute keys (as defined in the active app spec).
            reasoning_effort: Optional reasoning effort override.
            output_file: Optional output filename/path.
                - If relative, written under config paths.output_dir.
                - If absolute, written to that exact location.
            context_file: Optional context text file to inject as a leading system message
                for every pairwise comparison.
            context_date: Optional override date passed to prompt generation.

        Returns:
            Path to output CSV file.
        """
        if not real_profiles or len(real_profiles) < 2:
            raise ValueError("collect_allcomb requires at least 2 real profiles.")

        logger = get_experiment_logger("allcomb", f"n{len(real_profiles)}")
        self._log_run_config(logger)

        # Pre-format for prompt injection (same display formatting as makeup profiles)
        ids: List[str] = []
        formatted_profiles: List[Dict[str, Any]] = []
        for rid, profile in real_profiles:
            ids.append(str(rid))
            formatted_profiles.append(self._get_real_profile_formatted(rid, real_profile=profile))

        external_knowledge: Optional[List[Dict[str, str]]] = None
        if context_file:
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

        # Output path
        if output_file is None:
            output_file = f"allcomb_{len(real_profiles)}.csv"
        csv_path = get_output_path(output_file)
        total_pairs = len(real_profiles) * (len(real_profiles) - 1) // 2

        cols = [
            "model",
            "temperature",
            "pair_id",
            "pair",
            "prompt_variant",
            "prompt",
            "prompt_response",
            "chosen_profile",
            "chosen_profile_id",
            "nonchosen_profile_id",
        ]
        if self.logprobs_enabled:
            cols += ["prob_chosen", "prob_nochosen"]

        file_exists = csv_path.is_file()
        completed_pair_ids: set[int] = set()

        def _pair_key(a: str, b: str) -> tuple[str, str]:
            return tuple(sorted((str(a), str(b))))

        pair_key_to_pair_id: Dict[tuple[str, str], int] = {}
        pair_sequence: List[Tuple[int, int, int]] = []
        pair_id_cursor = 0
        for i in range(len(real_profiles)):
            for j in range(i + 1, len(real_profiles)):
                pair_sequence.append((pair_id_cursor, i, j))
                pair_key_to_pair_id[_pair_key(ids[i], ids[j])] = pair_id_cursor
                pair_id_cursor += 1

        if file_exists:
            try:
                header_df = pd.read_csv(csv_path, nrows=0)
                existing_cols = set(header_df.columns)
                usecols: List[str] = []
                if "pair_id" in existing_cols:
                    usecols.append("pair_id")
                if "chosen_profile_id" in existing_cols and "nonchosen_profile_id" in existing_cols:
                    usecols += ["chosen_profile_id", "nonchosen_profile_id"]

                if usecols:
                    existing = pd.read_csv(csv_path, usecols=usecols)
                    if "pair_id" in existing.columns:
                        completed_pair_ids.update(
                            pd.to_numeric(existing["pair_id"], errors="coerce")
                            .dropna()
                            .astype(int)
                            .tolist()
                        )
                    if "chosen_profile_id" in existing.columns and "nonchosen_profile_id" in existing.columns:
                        for _, row in existing.iterrows():
                            chosen_id = row.get("chosen_profile_id")
                            nonchosen_id = row.get("nonchosen_profile_id")
                            if pd.isna(chosen_id) or pd.isna(nonchosen_id):
                                continue
                            mapped_pair_id = pair_key_to_pair_id.get(_pair_key(str(chosen_id), str(nonchosen_id)))
                            if mapped_pair_id is not None:
                                completed_pair_ids.add(mapped_pair_id)

                completed_pair_ids = {pid for pid in completed_pair_ids if 0 <= pid < total_pairs}
                logger.info(
                    "Resuming allcomb: %s/%s pairs already completed in %s",
                    len(completed_pair_ids),
                    total_pairs,
                    csv_path,
                )
            except Exception:
                # If we can't parse existing output, do not attempt resume.
                completed_pair_ids = set()
                logger.warning("Unable to parse existing allcomb output for resume; processing from scratch.")

        # Main loop
        start_time = datetime.now()
        last_call_time = start_time
        logger.info(f"API session start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        num_variants = int(self.cfg.get("collection", "num_prompt_variants", default=10) or 10)
        if num_variants <= 0:
            num_variants = 10

        if len(completed_pair_ids) >= total_pairs:
            logger.info("All %s pairs are already completed. Nothing to run.", total_pairs)
            return csv_path

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()

            for pair_id, i, j in pair_sequence:
                if pair_id in completed_pair_ids:
                    continue

                labels = random_label_only()
                pair_ids = [ids[i], ids[j]]
                profiles = {
                    labels[0]: formatted_profiles[i],
                    labels[1]: formatted_profiles[j],
                }

                prompt_variant = pair_id % num_variants
                prompt_pair, prompt_labels = self._get_prompt_pair(labels, profiles, pair_id)
                prompt = get_prompt_variant(
                    prompt_variant,
                    prompt_pair,
                    prompt_labels,
                    date_override=context_date,
                )
                if external_knowledge is not None:
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
                chosen_profile_id = None
                nonchosen_profile_id = None
                if chosen_profile is not None and res in labels:
                    chosen_idx = labels.index(res)
                    chosen_profile_id = pair_ids[chosen_idx]
                    nonchosen_profile_id = pair_ids[1 - chosen_idx]

                row = {
                    "model": self.logprobs_model if self.logprobs_enabled else self.model,
                    "temperature": self.logprobs_temperature if self.logprobs_enabled else self.temperature,
                    "pair_id": pair_id,
                    "pair": profiles,
                    "prompt_variant": prompt_variant,
                    "prompt": prompt,
                    "prompt_response": res,
                    "chosen_profile": chosen_profile,
                    "chosen_profile_id": chosen_profile_id,
                    "nonchosen_profile_id": nonchosen_profile_id,
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

    # Backward-compatible alias (introduced briefly during development)
    def collect_realpairs(
        self,
        real_profiles: List[Tuple[str, Dict[str, Any]]],
        reasoning_effort: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Path:
        return self.collect_allcomb(
            real_profiles=real_profiles,
            reasoning_effort=reasoning_effort,
            output_file=output_file,
        )

    def collect_top(
        self,
        real_profile_id: str,
        n_top: Optional[int] = None,
        score_column: str = "MLP_score",
        reasoning_effort: Optional[str] = None,
        output_file: Optional[str] = None,
        real_profile: Optional[Dict[str, Any]] = None,
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
        real_profile = self._get_real_profile_formatted(
            real_profile_id,
            real_profile=real_profile,
        )

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
        real_profile: Optional[Dict[str, Any]] = None,
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

        real_profile_formatted = self._get_real_profile_formatted(
            real_profile_id,
            real_profile=real_profile,
        )

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
        real_profile: Optional[Dict[str, Any]] = None,
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
        real_profile = self._get_real_profile_formatted(
            real_profile_id,
            real_profile=real_profile,
        )

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
