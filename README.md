# Ramedas Reports Agent

Сервіс автоматичного збору, розпізнавання та сортування фотозвітності шкільних і дошкільних харчоблоків ТОВ «Рамедас Україна» (Viber Desktop, із готовністю до підключення Telegram та WhatsApp WAHA).

---

## Що робить агент

1. **Забирає фото з групових чатів Viber**: моніторить локальну папку `ViberDownloads`, куди десктопний клієнт Viber автоматично скачує всі нові фото з груп у фоновому режимі.
2. **Відсіює дублікати та нерелевантні фото**:
   - До виклику ШІ рахує `SHA-256` хеш кожного файлу та пропускає повторно надіслані фото без витрати токенів.
   - Відділяє випадкові фото (етикетки продуктів, стікери на упаковках тощо) у `storage/ignored_non_reports/`.
3. **Розпізнає документи через Vision LLM**:
   - Класифікує тип документа: `act` (Акт про надання послуг харчування), `daily_report` (Щоденний звіт), `menu` (Меню), `application` (Заявка), `raw_materials_calc` (Розрахунок сировини), `waybill_internal` (Накладна зі складу Луцьк), `waybill_external` (ТТН від зовнішніх постачальників, хліб тощо), `food_photo` (Фото контрольної порції).
   - Ігнорує юрособи групи компаній (*«Рамедас Україна», «Фондако», «Мистецтво Смаку», «Школярик», «Лерчія»*, київську юрадресу та *Склад Луцьк*) і шукає саме навчальний заклад.
4. **Багатофакторно визначає заклад із довідника (`data/facilities.json`)**:
   - За назвою школи/ліцею/гімназії/садка на бланку;
   - За селом/адресою доставки (для хлібних ТТН);
   - За прізвищем кухаря/директора (як зі статичного довідника, так і з таблиці `learned_contacts`, яку агент автоматично поповнює з чітких Актів та Меню);
   - За номером круглої печатки *«Для накладних №...»* (автоматично запам'ятовується за закладом у `stamp_registry`).
5. **Контролює якість розпізнавання рукописного тексту**:
   - Якщо почерк чіткий і рейтинг впевненості високий (`score >= 0.78`) — кладе файл у `storage/reports/<Громада>/<ID_Назва_закладу>/<YYYY-MM-DD>/`.
   - Якщо курсив нерозбірливий (`legibility = poor` або `score < 0.78` без додаткового підтвердження прізвищем кухаря чи печаткою) — відкладає файл у `storage/needs_review/<YYYY-MM-DD>/REVIEW_FAC-XXXX_...` для швидкої перевірки людиною.
6. **Підтримує гарячу заміну моделей (Hot-Reload) та резервні моделі (Fallback Chain)**:
   - Конфігурація моделей знаходиться у `config/llm_routing.json` (підтримуються Google Gemini, OpenAI, Anthropic Claude, OpenRouter, Ollama/vLLM).
   - При помилці `429` / `503` або таймауті основної моделі роутер автоматично перемикається на резервну модель.
   - Зміни у `config/llm_routing.json` та `.env` підхоплюються на льоту без перезапуску процесу.

---

## Структура проєкту

```text
ramedas-reports-agent/
├── config/
│   └── llm_routing.json       # Ланцюжок основних і резервних ШІ-моделей (hot-reload у рантаймі)
├── data/
│   ├── facilities.json        # Довідник закладів (385 закладів, контакти, аліаси)
│   └── facilities.csv         # CSV-копія довідника
├── src/
│   ├── llm/
│   │   ├── prompts.py         # Системний промпт та генератор компактного довідника
│   │   ├── providers.py       # Клієнти Gemini, OpenAI/OpenRouter, Anthropic
│   │   └── router.py          # Маршрутизатор моделей із Circuit Breaker та hot-reload
│   ├── connectors/
│   │   ├── base.py            # Базовий інтерфейс джерела медіа
│   │   ├── viber_local.py     # Сканер папки ViberDownloads + SHA-256 + групування в пакети
│   │   └── stubs.py           # Заготовки під Telegram та WhatsApp (WAHA)
│   ├── config.py              # Завантаження налаштувань із .env
│   ├── logger.py              # Ротоване логування в консоль та logs/agent.log
│   ├── models.py              # Pydantic-моделі даних
│   ├── catalog.py             # Пошуковий рушій по довіднику закладів
│   ├── storage.py             # SQLite-база стану, реєстр печаток, автовивчення прізвищ
│   ├── viber_watchdog.py      # Моніторинг і автоперезапуск процесу Viber.exe
│   ├── pipeline.py            # Конвеєр пакетної обробки
│   └── main.py                # Точка входу CLI
├── BACKLOG.md                 # Відкриті питання та план наступних доробок
├── .env.example               # Шаблон змінних середовища
├── pyproject.toml             # Конфігурація проєкту та залежностей
└── requirements.txt           # Список Python-пакетів
```

---

## Встановлення та перший запуск (Windows)

### 1. Клонування репозиторію
```powershell
git clone https://github.com/agoloveichuk/ramedas-reports-agent.git
cd ramedas-reports-agent
```

### 2. Створення віртуального середовища Python (3.11 / 3.12)

**Варіант А (рекомендований — через `uv`, автоматично завантажить Python 3.12):**
```powershell
pip install --user uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

**Варіант Б (через стандартний `venv`, якщо встановлений Python 3.11+):**
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### 3. Налаштування `.env`
Скопіюйте шаблон `.env.example` у `.env` та вкажіть свій ключ API і шлях до папки `ViberDownloads`:
```powershell
Copy-Item .env.example .env
```

Відкрийте `.env` у редакторі:
```ini
GEMINI_API_KEY=ваш_ключ_google_ai_studio
GEMINI_MODEL=models/gemini-flash-lite-latest

# Шлях до папки, куди Viber Desktop зберігає фото з чатів:
VIBER_DOWNLOADS_DIR=C:/Users/<Ваше_Ім'я_Користувача>/OneDrive/Documents/ViberDownloads
# або C:/Users/<Ваше_Ім'я_Користувача>/Documents/ViberDownloads

FACILITIES_JSON_PATH=./data/facilities.json
STORAGE_ROOT=./storage
BATCH_WINDOW_SECONDS=45
CONFIDENCE_REVIEW_THRESHOLD=0.78
SILENT_MODE=true
```

---

## Як тестувати та запускати

### 1. Перевірка стану та конфігурації (`--status`)
Показує, чи запущений `Viber.exe`, який ланцюжок моделей активний, скільки закладів/прізвищ у базі та скільки нових фото лежить у черзі `ViberDownloads`:
```powershell
.\.venv\Scripts\python.exe -m src.main --status
```

### 2. Тестовий прогін на кількох фото без збереження (`--dry-run`)
Дозволяє перевірити розпізнавання на перших 5 фото з папки `ViberDownloads` без копіювання файлів і без запису в БД:
```powershell
.\.venv\Scripts\python.exe -m src.main --once --limit 5 --dry-run
```

### 3. Обробка обмеженої порції фото зі збереженням (`--once --limit N`)
Обробляє перші `N` нових фото, розкладає їх у `storage/` і записує результат у `storage/state.db`:
```powershell
.\.venv\Scripts\python.exe -m src.main --once --limit 20
```

### 4. Повний одноразовий прогін усієї черги (`--once`)
Обробляє всі нові (та раніше перервані) фото з `ViberDownloads`:
```powershell
.\.venv\Scripts\python.exe -m src.main --once
```

### 5. Постійний фоновий режим моніторингу (`--watch`)
Безперервно стежить за папкою `ViberDownloads`, чекає стабілізації пакета (`BATCH_WINDOW_SECONDS=45`), автоматично перевіряє та перезапускає процес `Viber.exe` кожні 30 секунд у разі його закриття:
```powershell
.\.venv\Scripts\python.exe -m src.main --watch
```

---

## Де дивитися результати та логи

Після запуску створюються папки:
- `storage/reports/<Громада>/<ID_Заклад>/<YYYY-MM-DD>/` — чистові звіти з високим рейтингом впевненості.
- `storage/needs_review/<YYYY-MM-DD>/` — бланки з нерозбірливим курсивом (у назві файлу вказано попередньо визначений заклад `REVIEW_FAC-XXXX_...`).
- `storage/unmatched/<YYYY-MM-DD>/` — фото без реквізитів закладу (наприклад, одиночні фото страв без бланків).
- `storage/ignored_non_reports/<YYYY-MM-DD>/` — відсіяні нерелевантні фото (етикетки продуктів тощо).
- `storage/state.db` — локальна база SQLite з історією обробки (`processed_media`), реєстром печаток (`stamp_registry`) та вивченими прізвищами (`learned_contacts`).
- `logs/agent.log` — ротований текстовий лог усіх подій та помилок.

---

## Зміна ШІ-провайдера або моделі на ходу

Відкрийте `config/llm_routing.json`. Ви можете змінювати порядок маршрутів, вмикати/вимикати їх (`"enabled": true/false`) або додавати ключі `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` у `.env` — запущений у режимі `--watch` агент підхопить зміни автоматично на наступному фото без перезапуску.
