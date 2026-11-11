"""Детерминированная проверка фактов и формы ответа, точечная починка."""
from __future__ import annotations

import re

import policy

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

FORBIDDEN = [
    (re.compile(r"прям\w*\s+(?:выход|участие|доступ)[^.]{0,45}750", re.I),
     "Нельзя смешивать порог доступа (750 кВт) и порог прямого участия (20 МВт)."),
    (re.compile(r"95\s*(?:%|процент)[^.]{0,35}(вод|штрих)", re.I),
     "95% относится только к электроэнергии; вода и штрих-коды — более 90%."),
    (re.compile(r"обслуживани\w*\s+не\s+треб", re.I),
     "Нельзя утверждать, что станция не требует обслуживания."),
    (re.compile(r"гарантируем[^.]{0,35}(экономи|окупаем|срок|выработк)", re.I),
     "Нельзя гарантировать экономию, окупаемость и сроки."),
    (re.compile(r"от\s+5\s*кВт|до\s+сотен\s+мегаватт", re.I),
     "Типовой диапазон солнечных станций — от 15 до 500 кВт."),
    (re.compile(r"(?:установлен\w*|подтверждён\w*|доказан\w*)\s+факт\s+хищени", re.I),
     "Результат поиска потерь вероятностный, факт устанавливает выездная проверка."),
    (re.compile(r"автоном\w*[^.]{0,30}сетев\w*\s+станци", re.I),
     "Сетевая станция не обеспечивает автономию при отключении сети."),
]

MTS_ROLE = re.compile(r"МТС[^.]{0,40}(участник\w*\s+оптов|энергосбытов\w*\s+компан|поставщик\w*\s+электроэнерг)", re.I)
NEGATION = re.compile(r"не\s+(является|выступает|становится|продаёт|продаем|продаёт)", re.I)


def _nums(text: str) -> set[str]:
    flat = text.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    return set(NUM_RE.findall(flat))


def check(answer: str, context: str, question: str) -> list[str]:
    issues: list[str] = []
    if not answer or len(answer) < 25:
        return ["Ответ пустой или слишком короткий."]
    for sent in policy.sentences(answer):
        for pattern, message in FORBIDDEN:
            if pattern.search(sent) and message not in issues:
                issues.append(message)
        if MTS_ROLE.search(sent) and not NEGATION.search(sent):
            issues.append("МТС не является участником оптового рынка, сбытовой компанией или поставщиком.")
    allowed = _nums(context) | _nums(question) | {str(y) for y in range(2018, 2036)} | {str(i) for i in range(0, 13)}
    bad = sorted(n for n in _nums(answer) if n not in allowed)
    if bad:
        issues.append("Числа не подтверждены материалами: " + ", ".join(bad[:6]) + ". Убери их или замени формулировкой из материалов.")
    if policy.CONTACT_RE.search(answer):
        issues.append("Нельзя просить контакты или предлагать оформить заявку.")
    if policy.word_count(answer) > 130:
        issues.append("Ответ длиннее 130 слов, сократи до 35–80.")
    return issues


def repair_messages(messages: list[dict], answer: str, issues: list[str]) -> list[dict]:
    fix = ("Перепиши свой ответ, устранив замечания. Сохрани формат: 3–4 предложения, 35–80 слов, "
           "одна строка, прямой ответ первым предложением, при необходимости один уточняющий вопрос "
           "в конце. Верни только текст ответа.\n\nТвой ответ:\n" + answer +
           "\n\nЗамечания:\n" + "\n".join("- " + i for i in issues))
    return messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": fix}]


def strip_bad_sentences(answer: str) -> str:
    kept = []
    for sent in policy.sentences(answer):
        bad = any(p.search(sent) for p, _ in FORBIDDEN)
        if MTS_ROLE.search(sent) and not NEGATION.search(sent):
            bad = True
        if policy.CONTACT_RE.search(sent):
            bad = True
        if not bad:
            kept.append(sent)
    return " ".join(kept).strip()