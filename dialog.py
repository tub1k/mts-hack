"""Dialogue parsing, retrieval query construction, and grounded customer facts."""
from __future__ import annotations

import json
import re

ROLE_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(пользователь|клиент|user|вопрос|ассистент|бот|консультант|assistant|bot)\s*[:：-]\s*",
    re.I | re.M,
)
USER_ROLES = {"пользователь", "клиент", "user", "вопрос"}

TOPIC_KEYWORDS = {
    "solar": [r"солнеч", r"панел", r"инвертор", r"кровл", r"крыш", r"скат", r"шифер", r"фасад",
              r"накопител", r"гибрид", r"автоном", r"инсоляц", r"выработк", r"производств.*электроэнерг",
              r"окупаем", r"фотоэлектр", r"резерв.*отключ", r"\bсэс\b"],
    "orem": [r"оптов", r"\bорэм\b", r"\bорем\b", r"сбытов.*надбавк", r"гарантирующ",
             r"\bгтп\b", r"балансирующ", r"точк.*поставк", r"750\s*квт", r"20\s*мвт"],
    "askue": [r"аскуэ", r"платформ.*учет", r"облачн", r"удаленн.*сбор", r"цех", r"арендатор",
              r"абонент", r"личн.*кабинет", r"успд", r"концентратор", r"симк", r"\bsim\b",
              r"счетчик", r"счётчик", r"прибор.*учет", r"техническ.*учет", r"поверк",
              r"сбор.*показан", r"магазин", r"торгов.*точк"],
    "photo": [r"по фото", r"фотограф", r"снимок", r"telegram", r"телеграм", r"распозна",
              r"штрих", r"барабан", r"дисплей", r"сняти.*показан", r"архив.*фото",
              r"год.*выпуск", r"многотариф"],
    "losses": [r"хищен", r"потер", r"воров", r"вмешательств", r"магнит", r"незаконн.*подключ",
               r"небаланс", r"выездн.*проверк", r"сетев.*компан", r"архив.*(данн|учет)"],
    "price": [r"ценов.*категор", r"одноставочн", r"двухставочн", r"зон.*суток", r"день.*ночь",
              r"почасов.*планирован", r"треть.*категор", r"четверт.*категор", r"пят.*категор", r"\bчетв[её]рт\w*\b", r"\bчетверт\w*\b"],
    "tariffs": [r"счет.*вырос", r"счёт.*вырос", r"подорожа", r"рост.*счет", r"тариф", r"переплач",
                r"плат.*мощност", r"средн.*цен", r"квитанц", r"сумм.*заметн", r"расход.*вырос"],
    "thresholds": [r"мощност", r"\bквт\b", r"\bмвт\b", r"напряжен", r"класс.*точност",
                   r"трансформатор", r"реактивн", r"\b670\b", r"\b150\s*квт\b"],
    "metering": [r"уч[её]т", r"двухтариф", r"интервальн", r"почасов", r"показан"],
    "micro": [r"микрогенерац", r"излиш", r"отдач.*сеть", r"продаж.*сеть"],
    "requests": [r"сколько.*сто", r"стоимост", r"расч[её]т", r"оценк.*(стоимост|проект|эконом)",
                 r"окупаем", r"выгод", r"невыгод", r"предварительн"],
    "general": [r"мтс", r"компан", r"регион", r"безопасн", r"договор", r"оферт",
                r"бесплатн", r"заявк", r"персональн.*данн", r"телефон", r"контакт"],
    "regulation": [r"закон", r"постановлен", r"\bфз\b", r"регулятор", r"штраф"],
    "faq": [r"реактивн", r"небаланс", r"\bгтп\b", r"кто.*отвеча"],
}

REGION_PATTERNS = [
    r"камчат", r"магадан", r"сахалин", r"чукот", r"краснодар", r"ростов", r"астрахан",
    r"ставропол", r"москв", r"петербург", r"башкорт", r"татарстан", r"новосибирск",
    r"екатеринбург", r"казан", r"самар", r"челябинск", r"омск", r"перм",
    r"воронеж", r"волгоград", r"красноярск", r"иркутск", r"хабаровск", r"тюмен",
    r"калининград", r"мурманск", r"архангельск", r"ленинградск", r"сибир", r"урал",
]
OBJECTS = [("производств", "производство"), ("завод", "завод"), ("склад", "склад"),
           ("магазин", "магазин"), ("торгов", "торговый объект"),
           ("бизнес-центр", "бизнес-центр"), ("снт", "СНТ"), ("тсж", "ТСЖ"),
           ("сетев компан", "сетевая компания"), ("гостиниц", "гостиница"),
           ("офис", "офис"), ("арендатор", "арендаторы")]


def norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def _parse_json(raw: str):
    if not raw.startswith(("[", "{")):
        return None
    try:
        data = json.loads(raw)
        turns = data.get("messages", []) if isinstance(data, dict) else data
        turns = [("user" if v.get("role", "").lower() in ("user", "пользователь", "клиент") else "assistant",
                  str(v.get("content", ""))) for v in turns]
        return [(r, t.strip()) for r, t in turns if t.strip()]
    except (ValueError, TypeError, AttributeError):
        return None


def parse(text: str) -> dict:
    raw = (text or "").strip()
    turns = _parse_json(raw)
    if turns is None:
        marks = list(ROLE_RE.finditer(raw))
        turns = []
        if len(marks) >= 2:
            for i, m in enumerate(marks):
                end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
                body = raw[m.end():end].strip()
                if body:
                    turns.append(("user" if norm(m.group(1)) in USER_ROLES else "assistant", body))
    if not turns:
        return {"question": raw, "history": [], "is_dialog": False,
                "user_text": raw, "routing_text": raw, "search_query": raw}
    last_user_pos = next((i for i in range(len(turns)-1, -1, -1) if turns[i][0] == "user"), -1)
    if last_user_pos < 0:
        return {"question": raw, "history": [], "is_dialog": False,
                "user_text": raw, "routing_text": raw, "search_query": raw}
    question = turns[last_user_pos][1]
    history = turns[:last_user_pos]
    user_text = "\n".join(body for role, body in turns[:last_user_pos + 1] if role == "user")
    previous_user = [body for role, body in history if role == "user"]
    assistant_hint = next((body for role, body in reversed(history) if role == "assistant"), "")
    # Retrieval needs antecedents for elliptical follow-ups ("таких данных хватит?").
    # Repeat the current question once for relevance; include at most two prior user turns.
    search_query = "\n".join([question, question, assistant_hint] + previous_user[-3:])[:1900]
    return {"question": question, "history": history[-8:], "is_dialog": True,
            "user_text": user_text, "routing_text": user_text + "\n" + assistant_hint,
            "search_query": search_query}


def extract_facts(user_text: str) -> list[str]:
    low = norm(user_text)
    found = []
    for m in re.finditer(r"(?<!\w)(\d[\d\s.,]{0,12})\s*(мвт|квт)\b(?!\s*(?:[·*]?\s*)?ч)", low):
        found.append("Мощность, указанная клиентом: " + m.group(1).strip().rstrip(".,") + " " + m.group(2))
        break
    for m in re.finditer(r"(\d[\d\s.,]{0,12})\s*(?:квт\s*[·*]?\s*ч|квтч)", low):
        found.append("Потребление, указанное клиентом: " + m.group(0))
        break
    for p in REGION_PATTERNS:
        m = re.search(r"\b[а-я]*" + p + r"[а-я]*", low)
        if m:
            found.append("Регион упомянут: " + m.group(0))
            break
    for sub, label in OBJECTS:
        if sub in low:
            found.append("Объект: " + label)
    for sub, label in [("шифер", "шиферная кровля"), ("фасад", "фасад"),
                       ("скат", "скатная крыша"), ("плоск", "плоская крыша")]:
        if sub in low:
            found.append("Размещение: " + label)
    for sub, label in [("вечер", "вечернее потребление"), ("ночь", "ночная нагрузка"),
                       ("днем", "дневная нагрузка"), ("днём", "дневная нагрузка"),
                       ("круглосуточ", "круглосуточная нагрузка")]:
        if sub in low:
            found.append("Профиль потребления: " + label)
    for sub, label in [("почасов", "почасовые данные"), ("суточн", "суточные данные"),
                       ("двухтариф", "двухтарифные приборы"), ("аскуэ", "система АСКУЭ"),
                       ("фото", "показания по фото"), ("вручную", "ручной сбор")]:
        if sub in low:
            found.append("Учёт: " + label)
    for phrase, label in [("два года", "2 года"), ("двух лет", "2 года"),
                          ("за год", "1 год"), ("сорока счетчик", "около 40 счётчиков")]:
        if phrase in low:
            found.append("Указанный объём/архив: " + label)
    cat = re.search(r"(перв|втор|треть|четверт|пят|шест)\w*\s+(?:ценов\w*\s+)?категор", low)
    if cat:
        found.append("Ценовая категория упомянута: " + cat.group(0))
    for m in re.finditer(r"\b(\d+)\s*(магазин\w*|склад\w*|счетчик\w*|точ\w*|прибор\w*)", low):
        found.append("Количество: " + m.group(0))
    seen = set()
    result = []
    for f in found:
        if f not in seen:
            result.append(f)
            seen.add(f)
    return result[:12]


def route_topics(question: str, user_text: str) -> list[str]:
    current, context = norm(question), norm(user_text)
    scores = {}
    for topic, pats in TOPIC_KEYWORDS.items():
        current_score = sum(1 for p in pats if re.search(p, current))
        context_score = sum(1 for p in pats if re.search(p, context))
        # Specific product names outrank generic words like "учёт" / "компания".
        priority = {"orem": 2.3, "solar": 2.3, "photo": 2.2, "losses": 2.1,
                    "price": 2.0, "tariffs": 1.8, "requests": 1.8,
                    "askue": 1.5, "faq": 1.4, "thresholds": 0.8,
                    "metering": 0.65, "general": 0.55}.get(topic, 1.0)
        score = (current_score * 2 + context_score * 0.55) * priority
        if score:
            scores[topic] = score
    return sorted(scores, key=lambda k: (-scores[k], k))[:5]


SYNONYMS = {
    "симк": "SIM карта оператор связи мобильный интернет совместимость приборов учёта",
    "счетчик": "прибор учёта счётчик показания",
    "счётчик": "прибор учёта счётчик показания",
    "крыша": "кровля размещение солнечных модулей",
    "панел": "солнечная электростанция фотоэлектрические модули",
    "фасад": "вертикальное размещение солнечных панелей",
    "оптов": "оптовый рынок электроэнергии ОРЭМ",
    "хищен": "коммерческие потери необаланс выездная проверка",
    "потер": "коммерческие потери обнаружение аномалий",
    "по фото": "распознавание показаний по фотографии OCR",
    "реактив": "компенсация реактивной мощности измерение",
    "ценовая категор": "цена электроэнергии стоимость мощности передача",
    "часы": "почасовое потребление профиль нагрузки",
    "вечер": "вечернее потребление солнечная станция накопители",
    "архив": "данные энергопотребления часовые суточные",
    "арендатор": "технический учёт отдельных абонентов приборы",
    "стоимост": "предварительный расчёт запрос исходные данные",
}


def expand_query(question: str) -> str:
    low = norm(question)
    extra = [v for sub, v in SYNONYMS.items() if sub in low]
    return question + (" " + " ".join(extra) if extra else "")
