"""
services/ai_service.py
Google Gemini-powered cluster analysis with graceful rule-based fallback.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


PREFERRED_MODELS = (
    "gemini-1.5-flash-latest"
)


@lru_cache(maxsize=8)
def _get_supported_generate_models(api_key: str) -> tuple[str, ...]:
    """Return model names that support generateContent for the given API key."""
    if not GEMINI_AVAILABLE or not api_key:
        return tuple()

    try:
        supported = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" not in methods:
                continue
            full_name = str(getattr(model, "name", ""))
            if full_name.startswith("models/"):
                supported.append(full_name.split("/", 1)[1])
        return tuple(supported)
    except Exception:
        return tuple()


def _resolve_model_name(api_key: str, requested_model: Optional[str]) -> Optional[str]:
    """Choose a valid model name with graceful fallback."""
    env_model = (os.getenv("GEMINI_MODEL") or "").strip()
    requested = (requested_model or "").strip()
    candidates = [m for m in [requested, env_model, *PREFERRED_MODELS] if m]

    available = _get_supported_generate_models(api_key)
    if available:
        for candidate in candidates:
            if candidate in available:
                return candidate
        for name in available:
            if "gemini" in name:
                return name
        if available:
            return available[0]

    if candidates:
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompt(cluster_id: int, stats: dict, all_profiles_summary: str) -> str:
    """Build per-cluster analysis prompt, truncating summary to avoid token overflow."""
    summary_lines = all_profiles_summary.splitlines()
    if len(summary_lines) > 30:
        all_profiles_summary = "\n".join(summary_lines[:30]) + "\n... (truncated)"

    stats_text = json.dumps(stats, indent=2, default=str)

    return (
        "You are a senior marketing analyst. Analyze this customer segment and return ONLY a JSON object.\n\n"
        "Other segments summary (for context):\n"
        f"{all_profiles_summary}\n\n"
        f"Cluster ID: {cluster_id}\n"
        "This cluster's mean feature values:\n"
        f"{stats_text}\n\n"
        "Return ONLY this JSON, no markdown, no extra text, no trailing commas:\n"
        "{\n"
        '  "segment_name": "Short name max 4 words",\n'
        '  "description": "2 sentences about who these customers are vs other segments.",\n'
        '  "behavior_insight": "One sentence unique behavioral trait.",\n'
        '  "marketing_strategy": "One sentence specific strategy.",\n'
        '  "suggested_campaigns": ["Campaign 1", "Campaign 2", "Campaign 3"]\n'
        "}"
    )


def _build_overall_prompt(all_profiles_summary: str, cluster_insights: dict) -> str:
    """Build cross-cluster overall analysis prompt."""
    segment_lines = []
    for cid, ins in cluster_insights.items():
        name = ins.get("segment_name", f"Cluster {cid}")
        strategy = ins.get("marketing_strategy", "")
        segment_lines.append(f"- Cluster {cid} ({name}): {strategy}")
    segments_summary = "\n".join(segment_lines)

    summary_lines = all_profiles_summary.splitlines()
    if len(summary_lines) > 30:
        all_profiles_summary = "\n".join(summary_lines[:30]) + "\n... (truncated)"

    return (
        "You are a senior business strategist. Given these customer segments, provide a "
        "cross-cluster analysis. Return ONLY a JSON object.\n\n"
        "Cluster profiles (mean feature values):\n"
        f"{all_profiles_summary}\n\n"
        "Segment overview:\n"
        f"{segments_summary}\n\n"
        "Return ONLY this JSON, no markdown, no extra text, no trailing commas:\n"
        "{\n"
        '  "cluster_comparison": [\n'
        '    {"aspect": "Spending Power", "summary": "One sentence comparing all clusters."},\n'
        '    {"aspect": "Engagement Level", "summary": "One sentence comparing all clusters."},\n'
        '    {"aspect": "Growth Potential", "summary": "One sentence comparing all clusters."}\n'
        "  ],\n"
        '  "key_contrast": "1-2 sentences on the most striking contrast between any two clusters.",\n'
        '  "overall_strategy": "2-3 sentences on a unified strategy leveraging all segments.",\n'
        '  "priority_actions": [\n'
        '    "Action 1 for highest-value opportunity.",\n'
        '    "Action 2 addressing biggest risk.",\n'
        '    "Action 3 for long-term growth."\n'
        "  ]\n"
        "}"
    )


# ---------------------------------------------------------------------------
# JSON repair helper
# ---------------------------------------------------------------------------

def _parse_json_safe(raw: str) -> dict:
    """
    Strip markdown fences then parse JSON.
    If Gemini truncated the response, repair by closing open structures.
    """
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # First attempt: parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Repair: walk backwards to find last clean closing character,
    # then close all unclosed brackets/braces
    for i in range(len(raw) - 1, -1, -1):
        ch = raw[i]
        if ch not in ("}", "]", '"') and not ch.isdigit():
            continue
        candidate = raw[: i + 1]
        opens: list[str] = []
        in_str = False
        escape = False
        for c in candidate:
            if escape:
                escape = False
                continue
            if c == "\\" and in_str:
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c in ("{", "["):
                opens.append("]" if c == "[" else "}")
            elif c in ("}", "]") and opens:
                opens.pop()
        closing = "".join(reversed(opens))
        try:
            return json.loads(candidate + closing)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Cannot repair JSON response: {raw[:200]}")


# ---------------------------------------------------------------------------
# Rule-based fallbacks
# ---------------------------------------------------------------------------

def _rule_based_insight(cluster_id: int, stats: dict) -> dict:
    """Simple rule-based insight when Gemini is unavailable."""
    values = [v for v in stats.values() if isinstance(v, (int, float))]
    avg = sum(values) / max(len(values), 1)

    if avg > 0.5:
        name = "High-Value Customers"
        desc = "This segment shows above-average metrics across most features, indicating engaged and high-value customers."
        behavior = "Frequent purchasers with strong brand loyalty."
        strategy = "Retention-focused campaigns with VIP perks and loyalty rewards."
        campaigns = ["VIP Loyalty Program", "Premium Upsell Campaign", "Referral Reward Program"]
    elif avg < -0.3:
        name = "At-Risk Customers"
        desc = "This segment shows below-average metrics, suggesting disengaged or churning customers."
        behavior = "Low engagement with infrequent interactions."
        strategy = "Re-engagement campaigns with discounts and win-back offers."
        campaigns = ["Win-Back Email Series", "Discount Voucher Campaign", "Re-Engagement Survey"]
    else:
        name = "Mid-Tier Customers"
        desc = "This segment represents typical customers with average engagement and spend."
        behavior = "Moderate purchasing frequency with potential for upselling."
        strategy = "Cross-sell and educational campaigns to increase engagement."
        campaigns = ["Cross-Sell Bundle Campaign", "Educational Content Series", "Seasonal Promotions"]

    return {
        "segment_name": f"{name} (Cluster {cluster_id})",
        "description": desc,
        "behavior_insight": behavior,
        "marketing_strategy": strategy,
        "suggested_campaigns": campaigns,
    }


def _rule_based_overall(cluster_insights: dict) -> dict:
    """Rule-based overall analysis when Gemini is unavailable."""
    n = len(cluster_insights)

    comparison = [
        {
            "aspect": "Segment Size & Diversity",
            "summary": (
                f"The dataset contains {n} distinct customer segments, each with unique "
                "behavioral and demographic profiles that require tailored approaches."
            ),
        },
        {
            "aspect": "Engagement Levels",
            "summary": (
                "Segments vary significantly in engagement — high-value clusters show frequent "
                "interactions while at-risk clusters show declining activity."
            ),
        },
        {
            "aspect": "Revenue Potential",
            "summary": (
                "High-value segments contribute disproportionately to revenue, while mid-tier "
                "segments represent the largest growth opportunity through targeted upselling."
            ),
        },
    ]

    key_contrast = (
        "The most striking contrast is between the high-value loyal customers and the at-risk "
        "disengaged segment — their diverging behaviors demand completely different investment strategies."
    )

    overall_strategy = (
        f"With {n} distinct customer segments identified, the business should adopt a tiered "
        "engagement model: prioritise retention for high-value customers, activate growth campaigns "
        "for mid-tier segments, and deploy targeted re-engagement programmes for at-risk customers. "
        "Allocating budget proportionally to lifetime value potential across segments will maximise overall ROI."
    )

    priority_actions = [
        "Launch a VIP retention programme for the highest-value segment to protect core revenue.",
        "Develop a structured upsell funnel for mid-tier customers to accelerate their progression to high-value status.",
        "Implement an automated win-back sequence for at-risk segments before churn becomes irreversible.",
    ]

    return {
        "cluster_comparison": comparison,
        "key_contrast": key_contrast,
        "overall_strategy": overall_strategy,
        "priority_actions": priority_actions,
    }


# ---------------------------------------------------------------------------
# Main analysis functions
# ---------------------------------------------------------------------------

def analyze_cluster(
    cluster_id: int,
    stats: dict,
    all_profiles_summary: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict:
    """
    Analyze a single cluster using Google Gemini.
    Falls back to rule-based logic if API key is missing or call fails.

    Args:
        cluster_id: Integer cluster label.
        stats: Dict of {feature_name: mean_value}.
        all_profiles_summary: String summary of all cluster profiles for context.
        api_key: Gemini API key (falls back to env var GEMINI_API_KEY).
        model_name: Gemini model name override.

    Returns:
        Dict with: segment_name, description, behavior_insight,
                   marketing_strategy, suggested_campaigns.
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")

    if not key or not GEMINI_AVAILABLE:
        return _rule_based_insight(cluster_id, stats)

    try:
        genai.configure(api_key=key)
        resolved_model = _resolve_model_name(key, model_name)
        if not resolved_model:
            print("[AI Service] No valid Gemini model found. Falling back to rules.")
            return _rule_based_insight(cluster_id, stats)

        model = genai.GenerativeModel(
            model_name=resolved_model,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=2048,
            ),
        )
        response = model.generate_content(_build_prompt(cluster_id, stats, all_profiles_summary))
        return _parse_json_safe(response.text.strip())

    except Exception as exc:
        print(f"[AI Service] Gemini error for cluster {cluster_id}: {exc}")
        return _rule_based_insight(cluster_id, stats)


def analyze_overall(
    profiles: "pd.DataFrame",
    cluster_insights: dict,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict:
    """
    Generate a cross-cluster comparison and overall business strategy using Gemini.
    Falls back to rule-based logic if API key is missing or call fails.

    Args:
        profiles: DataFrame of cluster mean profiles (index = cluster_id).
        cluster_insights: Dict of {cluster_id: insight_dict} from analyze_all_clusters.
        api_key: Gemini API key (falls back to env var GEMINI_API_KEY).
        model_name: Gemini model name override.

    Returns:
        Dict with: cluster_comparison, key_contrast, overall_strategy, priority_actions.
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")

    if not key or not GEMINI_AVAILABLE:
        return _rule_based_overall(cluster_insights)

    try:
        genai.configure(api_key=key)
        resolved_model = _resolve_model_name(key, model_name)
        if not resolved_model:
            print("[AI Service] No valid Gemini model found for overall analysis. Falling back to rules.")
            return _rule_based_overall(cluster_insights)

        model = genai.GenerativeModel(
            model_name=resolved_model,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=2048,
            ),
        )

        all_profiles_summary = profiles.to_string(float_format="%.2f")
        prompt = _build_overall_prompt(all_profiles_summary, cluster_insights)
        response = model.generate_content(prompt)
        return _parse_json_safe(response.text.strip())

    except Exception as exc:
        print(f"[AI Service] Gemini error for overall analysis: {exc}")
        return _rule_based_overall(cluster_insights)


def analyze_all_clusters(
    profiles: "pd.DataFrame",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict:
    """
    Analyze all clusters in a profile DataFrame.
    Returns {cluster_id: insight_dict}.
    """
    results = {}
    all_profiles_summary = profiles.to_string(float_format="%.2f")

    for cluster_id in profiles.index:
        stats = profiles.loc[cluster_id].to_dict()
        results[int(cluster_id)] = analyze_cluster(
            int(cluster_id), stats, all_profiles_summary, api_key, model_name
        )
    return results