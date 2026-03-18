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
        "Bạn là một chuyên gia phân tích marketing cao cấp. Phân tích phân khúc khách hàng này và trả về CHỈ một đối tượng JSON.\n"
        "QUAN TRỌNG: Tất cả nội dung phải được viết bằng tiếng Việt.\n\n"
        "Tóm tắt các phân khúc khác (để tham chiếu):\n"
        f"{all_profiles_summary}\n\n"
        f"Mã cụm: {cluster_id}\n"
        "Giá trị trung bình các đặc trưng của cụm này:\n"
        f"{stats_text}\n\n"
        "Trả về CHỈ JSON này, không markdown, không text thừa, không dấu phẩy cuối:\n"
        "{\n"
        '  "segment_name": "Tên ngắn tối đa 4 từ bằng tiếng Việt",\n'
        '  "description": "2 câu bằng tiếng Việt mô tả khách hàng trong phân khúc này so với các phân khúc khác.",\n'
        '  "behavior_insight": "Một câu bằng tiếng Việt về đặc điểm hành vi nổi bật.",\n'
        '  "marketing_strategy": "Một câu bằng tiếng Việt về chiến lược cụ thể.",\n'
        '  "suggested_campaigns": ["Chiến dịch 1 bằng tiếng Việt", "Chiến dịch 2", "Chiến dịch 3"]\n'
        "}"
    )


def _build_overall_prompt(all_profiles_summary: str, cluster_insights: dict) -> str:
    """Build cross-cluster overall analysis prompt."""
    segment_lines = []
    for cid, ins in cluster_insights.items():
        name = ins.get("segment_name", f"Cluster {cid}")
        strategy = ins.get("marketing_strategy", "")
        segment_lines.append(f"- Cụm {cid} ({name}): {strategy}")
    segments_summary = "\n".join(segment_lines)

    summary_lines = all_profiles_summary.splitlines()
    if len(summary_lines) > 30:
        all_profiles_summary = "\n".join(summary_lines[:30]) + "\n... (truncated)"

    return (
        "Bạn là một chiến lược gia kinh doanh cao cấp. Dựa trên các phân khúc khách hàng này, "
        "hãy cung cấp phân tích liên cụm. Trả về CHỈ một đối tượng JSON.\n"
        "QUAN TRỌNG: Tất cả nội dung phải được viết bằng tiếng Việt.\n\n"
        "Hồ sơ các cụm (giá trị trung bình đặc trưng):\n"
        f"{all_profiles_summary}\n\n"
        "Tổng quan phân khúc:\n"
        f"{segments_summary}\n\n"
        "Trả về CHỈ JSON này, không markdown, không text thừa, không dấu phẩy cuối:\n"
        "{\n"
        '  "cluster_comparison": [\n'
        '    {"aspect": "Sức mua", "summary": "Một câu bằng tiếng Việt so sánh tất cả các cụm."},\n'
        '    {"aspect": "Mức độ Tương tác", "summary": "Một câu bằng tiếng Việt so sánh tất cả các cụm."},\n'
        '    {"aspect": "Tiềm năng Tăng trưởng", "summary": "Một câu bằng tiếng Việt so sánh tất cả các cụm."}\n'
        "  ],\n"
        '  "key_contrast": "1-2 câu bằng tiếng Việt về sự tương phản nổi bật nhất giữa hai cụm bất kỳ.",\n'
        '  "overall_strategy": "2-3 câu bằng tiếng Việt về chiến lược thống nhất tận dụng tất cả các phân khúc.",\n'
        '  "priority_actions": [\n'
        '    "Hành động 1 bằng tiếng Việt cho cơ hội giá trị cao nhất.",\n'
        '    "Hành động 2 bằng tiếng Việt giải quyết rủi ro lớn nhất.",\n'
        '    "Hành động 3 bằng tiếng Việt cho tăng trưởng dài hạn."\n'
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
        name = "Khách hàng Giá trị cao"
        desc = "Phân khúc này có các chỉ số trên trung bình ở hầu hết đặc trưng, cho thấy khách hàng tích cực và có giá trị cao."
        behavior = "Mua hàng thường xuyên với mức trung thành thương hiệu cao."
        strategy = "Chiến dịch giữ chân khách hàng với đặc quyền VIP và phần thưởng trung thành."
        campaigns = ["Chương trình Khách hàng VIP", "Chiến dịch Nâng cấp Cao cấp", "Chương trình Giới thiệu Thưởng"]
    elif avg < -0.3:
        name = "Khách hàng Có nguy cơ"
        desc = "Phân khúc này có các chỉ số dưới trung bình, cho thấy khách hàng không tích cực hoặc có nguy cơ rời bỏ."
        behavior = "Mức độ tương tác thấp với tần suất tương tác không thường xuyên."
        strategy = "Chiến dịch tái tương tác với giảm giá và ưu đãi thu hút quay lại."
        campaigns = ["Chuỗi Email Thu hút Quay lại", "Chiến dịch Phiếu Giảm giá", "Khảo sát Tái tương tác"]
    else:
        name = "Khách hàng Tầm trung"
        desc = "Phân khúc này đại diện cho khách hàng điển hình với mức độ tương tác và chi tiêu trung bình."
        behavior = "Tần suất mua hàng vừa phải với tiềm năng bán thêm."
        strategy = "Chiến dịch bán chéo và giáo dục để tăng mức độ tương tác."
        campaigns = ["Chiến dịch Gói Bán chéo", "Chuỗi Nội dung Giáo dục", "Khuyến mãi Theo mùa"]

    return {
        "segment_name": f"{name} (Cụm {cluster_id})",
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
            "aspect": "Quy mô & Đa dạng Phân khúc",
            "summary": (
                f"Tập dữ liệu chứa {n} phân khúc khách hàng riêng biệt, mỗi phân khúc có "
                "hồ sơ hành vi và nhân khẩu học riêng, đòi hỏi các phương pháp tiếp cận phù hợp."
            ),
        },
        {
            "aspect": "Mức độ Tương tác",
            "summary": (
                "Các phân khúc khác nhau đáng kể về mức độ tương tác — cụm giá trị cao có tương tác "
                "thường xuyên trong khi cụm có nguy cơ cho thấy hoạt động giảm sút."
            ),
        },
        {
            "aspect": "Tiềm năng Doanh thu",
            "summary": (
                "Phân khúc giá trị cao đóng góp tỷ trọng lớn vào doanh thu, trong khi phân khúc "
                "tầm trung đại diện cho cơ hội tăng trưởng lớn nhất thông qua bán thêm có mục tiêu."
            ),
        },
    ]

    key_contrast = (
        "Sự tương phản nổi bật nhất là giữa khách hàng trung thành giá trị cao và phân khúc "
        "không tích cực có nguy cơ — hành vi phân kỳ của họ đòi hỏi chiến lược đầu tư hoàn toàn khác nhau."
    )

    overall_strategy = (
        f"Với {n} phân khúc khách hàng riêng biệt được xác định, doanh nghiệp nên áp dụng mô hình "
        "tương tác theo tầng: ưu tiên giữ chân khách hàng giá trị cao, kích hoạt chiến dịch tăng trưởng "
        "cho phân khúc tầm trung, và triển khai chương trình tái tương tác có mục tiêu cho khách hàng có nguy cơ. "
        "Phân bổ ngân sách tỷ lệ thuận với tiềm năng giá trị vòng đời giữa các phân khúc sẽ tối đa hóa ROI tổng thể."
    )

    priority_actions = [
        "Triển khai chương trình giữ chân VIP cho phân khúc giá trị cao nhất để bảo vệ nguồn doanh thu cốt lõi.",
        "Xây dựng quy trình bán thêm có cấu trúc cho khách hàng tầm trung để đẩy nhanh sự chuyển đổi lên trạng thái giá trị cao.",
        "Thiết lập chuỗi thu hút quay lại tự động cho phân khúc có nguy cơ trước khi tỷ lệ rời bỏ trở nên không thể đảo ngược.",
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