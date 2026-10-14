"""
Report Generation Service — creates professional analysis reports for applications.
Generates formatted HTML, JSON, and markdown exports with full scoring explanation.
"""
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))

def generate_html_report(app_data: Dict[str, Any]) -> str:
    """
    Generate a professional HTML report for an application.
    Can be printed/saved as PDF from browser.
    """
    
    app = app_data
    company_name = "AgroSubsidia"
    report_date = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M:%S UTC")
    
    # Prepare data
    score = app.get("composite_score", 0)
    tier = app.get("tier", "—")
    criteria = app.get("criteria", [])
    explanation = app.get("explanation", "")
    full_context_explanation = app.get("full_context_explanation", "")
    extra_comment = app.get("llm_extra_comment", "")
    explanation_source = app.get("explanation_source", "template")
    sources = app.get("sources", [])
    rules = app.get("rules", {})
    
    # Color coding for tier
    tier_colors = {
        "Приоритет": "#10b981",
        "Рассмотреть": "#f59e0b",
        "Отклонить": "#ef4444",
    }
    tier_color = tier_colors.get(tier, "#6b7280")
    
    # Build criteria HTML
    criteria_html = ""
    for crit in criteria:
        if not crit or not crit.get("title"):
            continue
        
        status = crit.get("status", "")
        status_icon = "✓" if status == "ok" else "⚠" if status == "review" else "✗"
        status_color = "#10b981" if status == "ok" else "#f59e0b" if status == "review" else "#ef4444"
        
        score_val = crit.get("score_0_1")
        if score_val is None:
            score_val = crit.get("score_0_100", 0)
            score_disp = f"{score_val:.1f}/100"
        else:
            score_disp = f"{score_val:.2f}/1.0"

        points_awarded = crit.get("points_0_100")
        points_max = crit.get("max_points_0_100")
        if points_awarded is not None and points_max is not None and points_awarded != "N/A" and points_max != "N/A":
            try:
                points_disp = f"{float(points_awarded):.2f} / {float(points_max):.2f}"
            except ValueError:
                points_disp = "—"
        else:
            points_disp = "—"
        
        source_links = ""
        if crit.get("source_ids"):
            source_links = ", ".join(crit["source_ids"])
        
        criteria_html += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                <span style="font-size: 18px; color: {status_color};">{status_icon}</span>
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                <strong>{_escape_html(crit.get('title', 'N/A'))}</strong><br/>
                <small style="color: #6b7280;">{_escape_html(crit.get('explanation', ''))}</small>
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                <strong>{score_disp}</strong>
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                <strong>{points_disp}</strong>
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; font-size: 12px; color: #6b7280;">
                {_escape_html(source_links)}
            </td>
        </tr>
        """
    
    # Build sources HTML
    sources_html = ""
    for src in sources:
        if not src or not src.get("url"):
            continue
        sources_html += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">
                <strong>{_escape_html(src.get('title', 'N/A'))}</strong>
            </td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">
                <a href="{_escape_html(src.get('url', '#'))}" style="color: #3b82f6; text-decoration: none;">
                    {_escape_html(src.get('url', '—'))}
                </a>
            </td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; font-size: 12px;">
                {_escape_html(src.get('publisher', '—'))}
            </td>
        </tr>
        """
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчёт по скорингу заявки</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #1f2937;
            background: #f9fafb;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        header {{
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 32px;
            color: #0f172a;
            margin-bottom: 10px;
        }}
        .meta {{
            color: #6b7280;
            font-size: 14px;
        }}
        
        .score-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
            text-align: center;
        }}
        .score-value {{
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .score-tier {{
            font-size: 24px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 6px;
            display: inline-block;
        }}
        .score-tier.priority {{
            background: rgba(16, 185, 129, 0.3);
            color: #10b981;
        }}
        .score-tier.review {{
            background: rgba(245, 158, 11, 0.3);
            color: #f59e0b;
        }}
        .score-tier.reject {{
            background: rgba(239, 68, 68, 0.3);
            color: #ef4444;
        }}
        
        .explanation-box {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            margin: 30px 0;
            border-radius: 6px;
        }}
        .explanation-header {{
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .explanation-source {{
            font-size: 12px;
            display: inline-block;
            margin-top: 10px;
            padding: 4px 8px;
            border-radius: 4px;
            background: #dbeafe;
            color: #0369a1;
        }}
        .context-box {{
            background: #f8fafc;
            border-left: 4px solid #0ea5e9;
            padding: 20px;
            margin: 20px 0 30px;
            border-radius: 6px;
            white-space: pre-wrap;
        }}
        .subsection-box {{
            background: #f8fafc;
            padding: 18px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            margin-bottom: 16px;
        }}
        .subsection-box p {{
            margin: 6px 0;
        }}
        
        section {{
            margin: 40px 0;
        }}
        h2 {{
            font-size: 20px;
            color: #0f172a;
            margin-bottom: 20px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th {{
            background: #f3f4f6;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #d1d5db;
        }}
        
        .application-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        .info-block {{
            background: #f9fafb;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }}
        .info-block strong {{
            color: #0f172a;
            display: block;
            margin-bottom: 5px;
        }}
        .info-block span {{
            color: #6b7280;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            font-size: 12px;
            color: #6b7280;
            text-align: center;
        }}
        
        @media print {{
            body {{ background: white; padding: 0; }}
            .container {{ box-shadow: none; }}
            a {{ color: #0f172a; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Отчёт по скорингу заявки</h1>
            <div class="meta">
                <strong>Заявитель:</strong> {_escape_html(app.get('farmer_name', 'N/A'))}<br/>
                <strong>Направление:</strong> {_escape_html(app.get('direction', 'N/A'))}<br/>
                <strong>Дата отчёта:</strong> {report_date}
            </div>
        </header>
        
        <div class="score-card">
            <div>Композитный балл</div>
            <div class="score-value">{score:.1f}</div>
            <div class="score-tier {'priority' if tier == 'Приоритет' else 'review' if tier == 'Рассмотреть' else 'reject'}">
                {_escape_html(tier)}
            </div>
        </div>
        
        <div class="explanation-box">
            <div class="explanation-header">
                💡 Профессиональное объяснение
                <span class="explanation-source">
                    {'🤖 AI-анализ' if explanation_source == 'llm' else '⚙️ Автоматический'}
                </span>
            </div>
            <p>{_escape_html(explanation)}</p>
        </div>

        {f'''<div class="context-box"><strong>Развернутое NLP-пояснение</strong><br/><br/>{_escape_html(full_context_explanation)}</div>''' if full_context_explanation else ''}
        {f'''<div class="context-box" style="border-left-color:#10b981;"><strong>Комментарий по дополнительным пояснениям фермера</strong><br/><br/>{_escape_html(extra_comment)}</div>''' if extra_comment else ''}
        
        <section>
            <h2>📋 Данные по заявке</h2>
            <div class="application-info">
                <div class="info-block">
                    <strong>Направление животноводства</strong>
                    <span>{_escape_html(app.get('direction', 'N/A'))}</span>
                </div>
                <div class="info-block">
                    <strong>Категория субсидии</strong>
                    <span>{_escape_html(app.get('subsidy_category', 'N/A'))}</span>
                </div>
                <div class="info-block">
                    <strong>Запрашиваемая сумма</strong>
                    <span>{app.get('amount', 0):,.0f} ₸</span>
                </div>
                <div class="info-block">
                    <strong>Норматив</strong>
                    <span>{app.get('norm', 0):,.0f} ₸/голова</span>
                </div>
                <div class="info-block">
                    <strong>Поголовье</strong>
                    <span>{app.get('heads', 0):.1f} голов</span>
                </div>
                <div class="info-block">
                    <strong>Регион</strong>
                    <span>{_escape_html(app.get('region', 'N/A'))}</span>
                </div>
                <div class="info-block">
                    <strong>Район</strong>
                    <span>{_escape_html(app.get('district', 'N/A'))}</span>
                </div>
                <div class="info-block">
                    <strong>Дата подачи</strong>
                    <span>{_escape_html(app.get('date_submitted', 'N/A'))}</span>
                </div>
            </div>
        </section>

        <section>
            <h2>🧾 Полный контекст заявки</h2>
            <div class="subsection-box">
                <p><strong>Софинансирование:</strong> {_escape_html(app.get('own_contribution_percent', '—'))}%</p>
                <p><strong>Срок исполнения:</strong> {_escape_html(app.get('planned_execution_months', '—'))} мес.</p>
                <p><strong>Подтверждающие документы:</strong> {'Да' if app.get('has_supporting_documents') else 'Нет' if app.get('has_supporting_documents') is not None else '—'}</p>
                <p><strong>Цель заявки:</strong> {_escape_html(app.get('requested_purpose', '—'))}</p>
                <p><strong>Дополнительные пояснения:</strong> {_escape_html(app.get('extra_context', '—'))}</p>
            </div>
            <div class="subsection-box">
                <p><strong>Получено субсидий ранее:</strong> {_escape_html(app.get('previous_subsidies_count', '—'))}</p>
                <p><strong>Возвраты по субсидиям:</strong> {_escape_html(app.get('previous_subsidy_returns_count', '—'))}</p>
                <p><strong>Нарушения условий:</strong> {_escape_html(app.get('previous_subsidy_violations_count', '—'))}</p>
                <p><strong>Просрочки отчетности:</strong> {_escape_html(app.get('previous_subsidy_overdue_reports_count', '—'))}</p>
                <p><strong>Прошлые субсидии закрыты корректно:</strong> {'Да' if app.get('previous_subsidies_closed_properly') else 'Нет' if app.get('previous_subsidies_closed_properly') is not None else '—'}</p>
            </div>
        </section>
        
        <section>
            <h2>📊 Компоненты скоринга</h2>
            <div class="application-info">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px;">
                    <strong>ML Вероятность</strong><br/>
                    <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">
                        {app.get('ml_score', 0):.2%}
                    </div>
                    <span style="font-size: 12px; opacity: 0.9;">50% от общей оценки</span>
                </div>
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 8px;">
                    <strong>Правила</strong><br/>
                    <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">
                        {app.get('rule_score', 0):.2%}
                    </div>
                    <span style="font-size: 12px; opacity: 0.9;">30% от общей оценки</span>
                </div>
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 8px;">
                    <strong>Аномальность</strong><br/>
                    <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">
                        {('✓ Норма' if app.get('anomaly_clean') else '❌ Выброс')}
                    </div>
                    <span style="font-size: 12px; opacity: 0.9;">20% от общей оценки</span>
                </div>
            </div>
        </section>

        <section>
            <h2>✅ Базовые правила</h2>
            <div class="subsection-box">
                <p><strong>Поголовье ≥ 1:</strong> {'Да' if rules.get('rule_min_heads') == 1 else 'Нет'}</p>
                <p><strong>Норматив > 0:</strong> {'Да' if rules.get('rule_norm_positive') == 1 else 'Нет'}</p>
                <p><strong>Сумма > 0:</strong> {'Да' if rules.get('rule_amount_positive') == 1 else 'Нет'}</p>
                <p><strong>Сумма не экстремальна:</strong> {'Да' if rules.get('rule_not_extreme_amount') == 1 else 'Нет'}</p>
                <p><strong>Норматив в допустимом диапазоне:</strong> {'Да' if rules.get('rule_norm_in_range') == 1 else 'Нет'}</p>
            </div>
        </section>

        <section>
            <h2>🧠 SHAP-факторы</h2>
            <table>
                <thead>
                    <tr>
                        <th>Признак</th>
                        <th>SHAP-влияние</th>
                        <th>Сырое значение</th>
                        <th>Направление</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''<tr>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">{_escape_html(item.get('label', item.get('feature', '—')))}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">{_escape_html(item.get('shap_value', '—'))}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">{_escape_html(item.get('raw_value', '—'))}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">{_escape_html(item.get('direction', '—'))}</td>
                    </tr>''' for item in app.get('shap_explanations', [])) or '<tr><td colspan="4" style="padding: 12px; text-align: center; color: #6b7280;">SHAP-факторы отсутствуют</td></tr>'}
                </tbody>
            </table>
        </section>

        <section>
            <h2>📝 Решение комиссии</h2>
            <div class="subsection-box">
                <p><strong>Статус:</strong> {_escape_html(app.get('decision', 'pending'))}</p>
                <p><strong>Комментарий комиссии:</strong> {_escape_html(app.get('decision_note', '—'))}</p>
                <p><strong>Кто принял решение:</strong> {_escape_html(app.get('decision_by', '—'))}</p>
                <p><strong>Дата решения:</strong> {_escape_html(app.get('decision_at', '—'))}</p>
            </div>
        </section>
        
        <section>
            <h2>✓ Критерии оценивания</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 5%;">Статус</th>
                        <th style="width: 45%;">Критерий</th>
                        <th style="width: 15%;">Оценка</th>
                        <th style="width: 15%;">Баллы</th>
                        <th style="width: 20%;">Источники</th>
                    </tr>
                </thead>
                <tbody>
                    {criteria_html}
                </tbody>
            </table>
        </section>
        
        <section>
            <h2>📚 Источники данных</h2>
            <table>
                <thead>
                    <tr>
                        <th>Название источника</th>
                        <th>URL / Ссылка</th>
                        <th>Издатель</th>
                    </tr>
                </thead>
                <tbody>
                    {sources_html if sources_html else '<tr><td colspan="3" style="padding: 12px; text-align: center; color: #6b7280;">Источники не указаны</td></tr>'}
                </tbody>
            </table>
        </section>
        
        <div class="footer">
            <p>Этот отчёт был сгенерирован автоматической системой скоринга AgroSubsidia.</p>
            <p>На основе исторических данных 2024-2025 и официальных источников РК (Конституция, Stat.gov.kz, Data.egov.kz).</p>
            <p style="margin-top: 15px; font-size: 11px;">Версия: 2.5 | Распечатать: Ctrl+P</p>
        </div>
    </div>
</body>
</html>"""
    
    return html

def generate_json_report(app_data: Dict[str, Any]) -> str:
    """Generate a machine-readable JSON report."""
    
    report = {
        "metadata": {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "system_version": "2.5",
            "system_name": "AgroSubsidia Scoring System",
        },
        "application": {
            "id": app_data.get("id"),
            "farmer_name": app_data.get("farmer_name"),
            "date_submitted": app_data.get("date_submitted"),
            "decision": app_data.get("decision"),
            "decision_note": app_data.get("decision_note"),
            "decision_by": app_data.get("decision_by"),
            "decision_at": app_data.get("decision_at"),
        },
        "request": {
            "direction": app_data.get("direction"),
            "subsidy_name": app_data.get("subsidy_name"),
            "subsidy_category": app_data.get("subsidy_category"),
            "amount": app_data.get("amount"),
            "norm": app_data.get("norm"),
            "heads": app_data.get("heads"),
            "region": app_data.get("region"),
            "district": app_data.get("district"),
            "own_contribution_percent": app_data.get("own_contribution_percent"),
            "planned_execution_months": app_data.get("planned_execution_months"),
            "has_supporting_documents": app_data.get("has_supporting_documents"),
            "requested_purpose": app_data.get("requested_purpose"),
            "extra_context": app_data.get("extra_context"),
        },
        "subsidy_history": {
            "previous_subsidies_count": app_data.get("previous_subsidies_count"),
            "previous_subsidy_returns_count": app_data.get("previous_subsidy_returns_count"),
            "previous_subsidy_violations_count": app_data.get("previous_subsidy_violations_count"),
            "previous_subsidy_overdue_reports_count": app_data.get("previous_subsidy_overdue_reports_count"),
            "previous_subsidies_closed_properly": app_data.get("previous_subsidies_closed_properly"),
        },
        "scoring": {
            "composite_score": app_data.get("composite_score"),
            "tier": app_data.get("tier"),
            "ml_score": app_data.get("ml_score"),
            "rule_score": app_data.get("rule_score"),
            "anomaly_clean": app_data.get("anomaly_clean"),
            "rules": app_data.get("rules", {}),
        },
        "explanation": {
            "text": app_data.get("explanation"),
            "full_context_explanation": app_data.get("full_context_explanation"),
            "llm_extra_comment": app_data.get("llm_extra_comment"),
            "source": app_data.get("explanation_source"),
        },
        "criteria": app_data.get("criteria", []),
        "sources": app_data.get("sources", []),
        "shap_explanations": app_data.get("shap_explanations", []),
    }
    
    return json.dumps(report, ensure_ascii=False, indent=2)
