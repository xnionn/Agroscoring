"""
LLM Explain Service — generates natural-language scoring explanations via Groq API.
Falls back to template explanation if GROQ_API_KEY is not set or unavailable.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Load .env from project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_root, ".env"), override=False)
except ImportError:
    pass


def _get_groq_client():
    """Lazy Groq client init — returns None if key missing or package not installed."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        logger.warning("[LLM] groq package not installed. pip install groq")
        return None
    except Exception as exc:
        logger.warning(f"[LLM] Groq client init failed: {exc}")
        return None


def generate_llm_explanation(
    composite: float,
    tier: str,
    criteria: list,
    amount: float = 0,
    direction: str = "",
    region: str = "",
    profile_years: Optional[int] = None,
    profile_heads: Optional[int] = None,
    profile_land: Optional[float] = None,
    ml_score: Optional[float] = None,
    rule_score: Optional[float] = None,
    rules_detail: Optional[dict] = None,
    extra_context: Optional[str] = None,
) -> dict:
    """
    Generate a natural-language explanation in Russian for a scoring result.

    Returns:
        {
            "explanation": str | None,
            "full_context_explanation": str | None,
            "explanation_source": "llm" | "template",
            "extra_comment": str | None,
        }
    Falls back gracefully: if Groq is unavailable, returns explanation=None so
    the caller can use the existing template explanation.
    """
    client = _get_groq_client()
    if client is None:
        return {
            "explanation": None,
            "full_context_explanation": None,
            "explanation_source": "template",
            "extra_comment": None,
        }

    risky = [c for c in criteria if c.get("status") == "risk"]
    ok_items = [c for c in criteria if c.get("status") == "ok"]

    def _fmt_score(c):
        s = c.get("score_0_1")
        if s is None:
            s = c.get("score_0_100")
            if s is not None:
                return f"{s:.1f}/100"
            return "?"
        return f"{s:.2f}/1.0"

    risky_text = "\n".join(
        f"  • {c['title']} ({_fmt_score(c)}): {c.get('explanation', '')}"
        for c in risky
    ) or "  Не обнаружено"

    strengths_text = "\n".join(
        f"  • {c['title']} ({_fmt_score(c)}): {c.get('explanation', '')}"
        for c in ok_items[:5]
    ) or "  Нет выраженных сильных факторов"

    profile_lines = []
    if profile_years is not None:
        profile_lines.append(f"стаж {profile_years} лет")
    if profile_heads is not None:
        profile_lines.append(f"поголовье в профиле {profile_heads} голов")
    if profile_land is not None:
        profile_lines.append(f"земля {profile_land} га")
    profile_text = ", ".join(profile_lines) if profile_lines else "не указан"

    # Build detailed criteria breakdown for LLM context
    criteria_details = []
    for c in criteria:
        score_display = _fmt_score(c)
        weight = c.get("weight", 0)
        status_icon = "✓" if c.get("status") == "ok" else "✗"
        title = c.get("title", "?")
        explanation = c.get("explanation", "")
        value = c.get("value")
        
        detail = f"{status_icon} {title}: {score_display} (вес {weight*100:.0f}%)"
        if value:
            detail += f" — {value}"
        detail += f" · {explanation}"
        criteria_details.append(detail)
    
    criteria_text = "\n".join(criteria_details) if criteria_details else "Нет данных"
    # Add threshold information
    thresholds = ""
    if ml_score is not None:
        ml_threshold = 0.5
        ml_status = "✓ проходит пороговое значение" if ml_score >= ml_threshold else "✗ ниже порога"
        thresholds += f"\n  • ML вероятность >= {ml_threshold}: {ml_score:.2f} {ml_status}"
    
    if rule_score is not None:
        rule_threshold = 0.6
        rule_status = "✓ проходит пороговое значение" if rule_score >= rule_threshold else "✗ ниже порога"
        thresholds += f"\n  • Фиксированные правила >= {rule_threshold}: {rule_score:.2f} {rule_status}"

    extra_context_text = (extra_context or "").strip()
    extra_context_block = extra_context_text if extra_context_text else "не указано"

    # More specific and structured prompt focused on concrete facts
    prompt = (
        "Ты — финансовый аналитик по агрогосубсидиям Республики Казахстан. "
        "Без лишних слов, только конкретные факты и ссылки на данные. "
        "Верни ответ строго в трех секциях. "
        "Первая секция должна начинаться с EXPLANATION:, вторая с FULL_CONTEXT:, третья с EXTRA_COMMENT:. "
        "EXPLANATION — это короткий вывод в 2-4 предложениях. "
        "FULL_CONTEXT — это развернутое пояснение в 6-12 предложениях с полным контекстом оценки. "
        "В EXPLANATION обязательно назови не менее 3 самых важных факторов с цифрами или статусами критериев. "
        "В FULL_CONTEXT объясни сильные стороны, риски, влияние правил, профиля, документов и истории субсидий, если эти данные есть. "
        "EXTRA_COMMENT — короткий комментарий по свободному тексту фермера в 1-3 предложениях. "
        "Если свободный текст пустой, напиши EXTRA_COMMENT: нет дополнительного комментария. "
        "Если балл низкий — назови конкретные причины с цифрами. Если высокий — объясни, "
        "почему заявка соответствует требованиям. НЕ ИСПОЛЬЗУЙ markdown, НЕ ПИШИ общих фраз типа "
        "'требуется внимание комиссии' или 'рекомендуется рассмотреть'. Только факты и цифры.\n\n"
        f"=== ЗАЯВКА ===\n"
        f"Направление: {direction or 'не указано'}\n"
        f"Регион: {region or 'не указан'}\n"
        f"Сумма запрашивается: {int(amount):,} тг\n"
        f"Профиль: {profile_text}\n\n"
        f"=== СИЛЬНЫЕ ФАКТОРЫ ===\n{strengths_text}\n\n"
        f"=== РИСКИ ===\n{risky_text}\n\n"
        f"=== КРИТЕРИИ И ПОРОГИ ===\n"
        f"{criteria_text}{thresholds}\n\n"
        f"=== СВОБОДНЫЙ ТЕКСТ ФЕРМЕРА ===\n{extra_context_block}\n\n"
        f"=== ИТОГОВАЯ ОЦЕНКА ===\n"
        f"Балл: {composite:.1f}/100 → {tier}\n"
        f"Вывод: Опираясь на конкретные цифры выше, напиши прямое объяснение оценки и отдельно коротко оцени свободный текст.\n"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1,  # Lower temperature = more factual, less creative
            top_p=0.9,        # Conservative sampling
            frequency_penalty=0.2,  # Reduce repetition
        )
        text = response.choices[0].message.content.strip()
        explanation = None
        full_context_explanation = None
        extra_comment = None
        if "FULL_CONTEXT:" in text and "EXTRA_COMMENT:" in text:
            before_context, after_context = text.split("FULL_CONTEXT:", 1)
            full_context_block, extra_comment_block = after_context.split("EXTRA_COMMENT:", 1)
            if "EXPLANATION:" in before_context:
                explanation = before_context.split("EXPLANATION:", 1)[1].strip()
            full_context_explanation = full_context_block.strip()
            extra_comment = extra_comment_block.strip()
        else:
            for line in text.splitlines():
                normalized = line.strip()
                if normalized.startswith("EXPLANATION:"):
                    explanation = normalized.split("EXPLANATION:", 1)[1].strip()
                elif normalized.startswith("FULL_CONTEXT:"):
                    full_context_explanation = normalized.split("FULL_CONTEXT:", 1)[1].strip()
                elif normalized.startswith("EXTRA_COMMENT:"):
                    extra_comment = normalized.split("EXTRA_COMMENT:", 1)[1].strip()

        if not explanation:
            explanation = text
        if not full_context_explanation:
            full_context_explanation = explanation
        return {
            "explanation": explanation,
            "full_context_explanation": full_context_explanation,
            "explanation_source": "llm",
            "extra_comment": extra_comment,
        }
    except Exception as exc:
        logger.warning(f"[LLM] Groq request failed: {exc}")
        return {
            "explanation": None,
            "full_context_explanation": None,
            "explanation_source": "template",
            "extra_comment": None,
        }
