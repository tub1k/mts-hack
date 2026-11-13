"""Low-false-positive factual safety checks; repair only demonstrably problematic output."""
from __future__ import annotations

import re
import policy

BAD_PATTERNS = [
    (re.compile(r"прям\w*\s+(?:выход|участие|доступ)[^.]{0,55}\b750\s*квт", re.I),
     "Перепутаны пороги: 750 кВт — порог доступа, а прямое участие — от 20 МВт."),
    (re.compile(r"(?:95\s*%|95\s*процент)[^.]{0,65}(?:вод|штрих)|(?:вод|штрих)[^.]{0,65}(?:95\s*%|95\s*процент)", re.I),
     "95% относится к электроэнергии; для воды и штрихкодов указана другая точность."),
    (re.compile(r"(?:не\s+требует?|без)\s+(?:никакого\s+)?(?:обслуживани|ухода)", re.I),
     "Не следует обещать, что оборудование не требует обслуживания."),
    (re.compile(r"гарантир\w*[^.!?]{0,65}(?:экономи|окупаем|выработк|срок)", re.I),
     "Нельзя гарантировать экономию или окупаемость без расчёта."),
    (re.compile(r"(?:доказан|установлен|подтвержден)\w*[^.!?]{0,30}(?:хищени|воровств)", re.I),
     "Модель поиска потерь указывает кандидатов на проверку, а не доказанное хищение."),
    (re.compile(r"(?:сетев\w*\s+станци\w*)[^.!?]{0,65}(?:автономн\w*\s+работ|обеспеч\w*\s+резерв)[^.!?]{0,35}(?:без\s+накопител|при\s+отключени)", re.I),
     "Обычная сетевая станция без резервных компонентов не даёт автономию."),
]
CLAIMS_ROLE = re.compile(r"\bМТС\b[^.!?]{0,65}(?:участник\w*\s+оптов|энергосбытов\w*\s+компан|поставщик\w*\s+электроэнерг)", re.I)
NEGATION = re.compile(r"\bне\s+(?:является|выступает|становится|продаёт|продает)\b", re.I)
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?=\s*(?:%|квт|мвт|квт\W*ч|руб|лет|года?\b))", re.I)
TRUSTED = {"1", "2", "3", "4", "6", "7", "9", "15", "20", "25", "60", "90", "95", "500", "750"}


def check(answer: str, context: str, question: str) -> list[str]:
    if not (answer or "").strip():
        return ["Пустой ответ"]
    issues = []
    for s in policy.sentences(answer):
        for pattern, explanation in BAD_PATTERNS:
            if pattern.search(s) and explanation not in issues:
                issues.append(explanation)
        if CLAIMS_ROLE.search(s) and not NEGATION.search(s):
            issues.append("МТС нельзя без подтверждения называть энергосбытовой компанией или поставщиком.")
        if policy.CONTACT_REQUEST_RE.search(s):
            issues.append("Не запрашивай контакты и не предлагай оформить заявку.")
    # Only verify explicit domain numbers (with units); plain enumeration, years,
    # counts and customer-provided amounts should never trigger a false repair.
    referenced = {x.group(0).replace(",", ".") for x in NUMBER_RE.finditer(context + " " + question)}
    unsupported = []
    for match in NUMBER_RE.finditer(answer):
        n = match.group(0).replace(",", ".")
        if n not in referenced and n not in TRUSTED and n not in unsupported:
            unsupported.append(n)
    if unsupported:
        issues.append("Числовые утверждения с единицами измерения не подтверждены показанными источниками: "
                      + ", ".join(unsupported[:5]))
    if policy.word_count(answer) > 155:
        issues.append("Ответ превышает рекомендуемый максимум 150 слов.")
    return issues


def repair_messages(messages: list[dict], answer: str, issues: list[str]) -> list[dict]:
    task = ("Исправь фактические ошибки и нарушение границ консультации в своём ответе. "
            "Сохрани все правильные полезные сведения и ответы на последний вопрос. "
            "Дай 2–5 предложений, до 150 слов, без просьбы контактов. "
            "Не вводи новые факты, отсутствующие в материалах. Верни только исправленный текст.\n\n"
            "Исходный ответ:\n" + answer + "\n\nПроблемы:\n" +
            "\n".join("- " + issue for issue in issues))
    return messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": task}]


def strip_bad_sentences(answer: str) -> str:
    parts = []
    for sentence in policy.sentences(answer):
        if any(p.search(sentence) for p, _ in BAD_PATTERNS):
            continue
        if CLAIMS_ROLE.search(sentence) and not NEGATION.search(sentence):
            continue
        if policy.CONTACT_REQUEST_RE.search(sentence):
            continue
        parts.append(sentence)
    return " ".join(parts)
