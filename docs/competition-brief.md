# Hull Tactical — Market Prediction: что от нас требуется

Источник: https://www.kaggle.com/competitions/hull-tactical-market-prediction
Документ собран из вкладок Overview / Data, кода метрики и demo-submission ноутбука.

---

## 0. Статус (важно)

На странице соревнования сейчас стоит бейдж **Late Submission**. По официальному таймлайну:

| Событие | Дата |
|---|---|
| Старт | 16 сентября 2025 |
| Entry deadline / team merger | 8 декабря 2025 |
| Final submission deadline | 15 декабря 2025 |
| Конец forecasting-фазы | 25 июня 2026 |

То есть **зачётный цикл закрыт**: призы (50k / 25k / 10k / 5k×3) разыграны, попасть в лидерборд с призами уже нельзя.
Сабмиты возможны только как late submission — то есть проект имеет смысл как исследовательский / портфельный: строим модель, меряемся с бенчмарками офлайн на train-данных.

---

## 1. Задача в одном абзаце

Каждый торговый день на закрытии предсказать **не доходность, а долю капитала (allocation) в S&P 500**: число из отрезка **[0, 2]** (0 = всё в кэш под risk-free, 1 = полностью в индексе, 2 = двойное плечо).
Оценивается не MSE прогноза, а **риск-скорректированная доходность получившейся стратегии** — вариант Sharpe с двумя штрафами. То есть это одновременно задача прогноза excess returns *и* задача построения betting strategy под ограничение по волатильности (не более 120% волатильности рынка).

Идейная рамка от организаторов: проверка Efficient Market Hypothesis — можно ли машинным обучением найти устойчивый edge.

---

## 2. Данные

### `train.csv` — историческая рыночная выборка (десятилетия; в ранних годах много пропусков)

| Колонка | Что это |
|---|---|
| `date_id` | идентификатор торгового дня |
| `M*` | Market Dynamics / технические фичи |
| `E*` | Макроэкономические фичи |
| `I*` | Процентные ставки |
| `P*` | Цена / оценка (valuation) |
| `V*` | Волатильность |
| `S*` | Сентимент |
| `MOM*` | Моментум |
| `D*` | Dummy / бинарные |
| `forward_returns` | доходность «купил S&P 500 сегодня, продал завтра». **Только в train** |
| `risk_free_rate` | ставка ФРС (federal funds rate). **Только в train** |
| `market_forward_excess_returns` | forward returns относительно ожиданий: вычтено скользящее 5-летнее среднее forward returns, результат винзоризован по MAD с критерием 4. **Только в train** |

### `test.csv` — макет тестовой выборки

| Колонка | Что это |
|---|---|
| `date_id` | день |
| `[feature_name]` | те же фичи, что в train |
| `is_scored` | входит ли строка в расчёт метрики. В training-фазе `True` только для первых 180 строк |
| `lagged_forward_returns` | forward returns **с лагом в 1 день** |
| `lagged_risk_free_rate` | ставка с лагом в 1 день |
| `lagged_market_forward_excess_returns` | excess returns с лагом в 1 день |

Ключевое следствие: **в inference-времени таргета за сегодня нет**, есть только его лаггированная версия (то есть факт за вчера). Всё, что строим онлайн (обновление модели, оценка режима), должно опираться на lagged-колонки.

### Публичный лидерборд бессмыслен

Public LB = копия **последних 180 date_id из train**. Организаторы прямо пишут, что скоры на этой фазе не значат ничего. Единственный честный сигнал — собственная временная валидация.

### Forecasting-фаза

* Тестовая выборка собирается **после закрытия сабмитов**, размер сопоставим с 180 днями.
* API отдаёт данные от начала public-сета до конца private-сета — включая дни до дедлайна, которые **не скорятся**.
* Первый `date_id`, отдаваемый API, остаётся неизменным всю дорогу.

### `kaggle_evaluation/`

Файлы evaluation API. Внутренние данные не публикуются, но после окончания соревнования Hull обещали периодически выкладывать свой датасет на своём сайте — им разрешено пользоваться для собственной торговли.

---

## 3. Метрика — точный код и что из него следует

Официальная метрика: https://www.kaggle.com/code/metric/hull-competition-sharpe

```python
MIN_INVESTMENT = 0
MAX_INVESTMENT = 2

def score(solution, submission, row_id_column_name) -> float:
    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(...)
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(...)

    solution['strategy_returns'] = (
        solution['risk_free_rate'] * (1 - solution['position'])
        + solution['position'] * solution['forward_returns']
    )

    # Sharpe стратегии
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = strategy_excess_cumulative ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Рынок
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = market_excess_cumulative ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    # Штраф за волатильность
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2)
    vol_penalty = 1 + excess_vol

    # Штраф за отставание по доходности
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap ** 2) / 100

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)
```

### Разбор

**Доходность стратегии.** `strategy_returns = rf·(1−pos) + pos·fwd`, значит
`strategy_excess = strategy_returns − rf = pos · (fwd − rf)`.
Позиция — множитель на рыночный excess return. Никаких транзакционных издержек в метрике нет (но организаторы отдельно упоминают frictions как реальную проблему).

**Sharpe.** Числитель — *геометрическое* среднее дневного excess (через `prod`, не `mean`) → просадки и «пилу» метрика наказывает сама по себе. Знаменатель — std **полной** доходности стратегии, не excess. Годовое масштабирование `√252`.

**Штраф за волатильность.** `vol_penalty = 1 + max(0, vol_ratio − 1.2)`.
Односторонний и «бесплатный» до 120% волатильности рынка. Превышение штрафует *линейно*: вола 150% рынка → делитель 1.3.

**Штраф за отставание.** `return_gap` — отставание годовой геометрической excess-доходности от рыночной **в процентных пунктах**, штраф `1 + gap²/100` — квадратичный:

| Отставание от рынка, п.п. годовых | `return_penalty` |
|---|---|
| 0 | 1.0 |
| 5 | 1.25 |
| 10 | 2.0 |
| 20 | 5.0 |
| 30 | 10.0 |

Это самый жёсткий элемент метрики: «пересидеть в кэше» безопаснее не становится — низкая вола не даёт бонуса, а отставание от рынка убивает скор квадратично.

### Практические выводы (важно для дизайна стратегии)

1. **Базовая линия — константная позиция.** При `pos = c` (const) и mean, и std масштабируются на `c`, поэтому Sharpe ≈ Sharpe рынка. При этом:
   * `c ≥ 1` → нет штрафа за отставание (если рыночный excess положителен);
   * `c ≤ 1.2` → нет штрафа за волатильность.
   Значит **любая константа из [1.0, 1.2] даёт скор ≈ Sharpe рынка бесплатно** — это тот бенчмарк, который нужно бить. Всё, что скорит хуже, — хуже «ничегонеделания».
2. **Единственный источник альфы — тайминг**: поднимать позицию в режимах с высоким ожидаемым excess/низкой волой и опускать в обратных. Улучшать надо именно отношение `geo_mean_excess / std`.
3. **Асимметрия штрафов** велит держать среднюю экспозицию около 1.0–1.2 и «дышать» вокруг неё, а не уходить в 0 или 2 надолго.
4. **Хард-констрейнт**: `min(pred) ≥ 0` и `max(pred) ≤ 2` по **всей** сабмишн-выборке, иначе — ошибка, а не просто плохой скор. Клипать обязательно.
5. **Горизонт скоринга ~180 дней** → метрика очень шумная. Оптимизировать надо ожидание скора по многим окнам, а не одно значение.

---

## 4. Evaluation API — как устроен сабмит

Соревнование **code competition**: сабмитим не CSV, а ноутбук-сервер, который отдаёт предсказания по одному временному шагу. Скелет из demo-submission:

```python
import os
import pandas as pd
import polars as pl
import kaggle_evaluation.default_inference_server

def predict(test: pl.DataFrame) -> float:
    """Сюда — своя инференс-логика.
    Можно вернуть Pandas или Polars DataFrame, Polars предпочтительнее по скорости.
    Каждый батч предсказаний (кроме самого первого) должен быть возвращён в течение 5 минут
    после выдачи фич этого батча."""
    return 0.0

inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
```

Ограничения из комментариев API:

* `inference_server.serve()` должен быть вызван **в пределах 15 минут** после старта ноутбука, иначе gateway падает с ошибкой.
* Тяжёлую загрузку модели можно унести в **первый вызов `predict`** — у него нет обычного дедлайна ответа (в коде фигурируют «1 минута» для обычного вызова и «5 минут» на батч; закладываемся на самый строгий).
* Клиент работает в отдельном контейнере с доступом к скрытому тесту и отдаёт данные **пошагово по времени** → заглянуть вперёд физически нельзя.
* Опубликованные файлы соревнования доступны коду всегда.

### Требования к ноутбуку

* CPU ≤ 8 часов, GPU ≤ 8 часов (в forecasting-фазе лимит поднимается до **9 часов**).
* **Интернет отключён.**
* Внешние данные и предобученные модели можно — если они свободно и публично доступны.

---

## 5. Чек-лист: что от нас требуется

- [ ] Загрузить данные, собрать EDA: покрытие фич по годам, доля пропусков, распределение `forward_returns`, `risk_free_rate`, режимы волатильности.
- [ ] Реализовать **метрику один в один** как локальную функцию + бэктестер (обязательно, метрика неочевидна).
- [ ] Зафиксировать бенчмарки: `pos ≡ 1.0`, `pos ≡ 1.2`, buy&hold, простые vol-targeting правила. Всё остальное сравнивать с ними.
- [ ] Построить схему валидации: только walk-forward / purged time-series split, никакого random KFold. Оценивать скор на множестве 180-дневных окон, смотреть на распределение, а не на среднее.
- [ ] Разобраться с пропусками: ранние десятилетия почти пустые → решить, с какого `date_id` учить.
- [ ] Модель прогноза excess return (и, отдельно, прогноз волатильности — он нужен для sizing).
- [ ] Слой перевода прогноза в позицию: Kelly-подобный / vol-targeting sizing, клип в [0, 2], сглаживание, контроль средней экспозиции ≈1.0–1.2 и отношения воля стратегии / воля рынка ≤ 1.2.
- [ ] Проверить онлайн-логику: в инференсе доступны только `lagged_*` таргеты; никаких обращений к `forward_returns` за текущий день.
- [ ] Собрать сабмит-ноутбук по скелету выше; проверить локально через `run_local_gateway`, уложиться в лимиты по времени, работать без интернета.

---

## 6. Ссылки

* Соревнование: https://www.kaggle.com/competitions/hull-tactical-market-prediction
* Данные: https://www.kaggle.com/competitions/hull-tactical-market-prediction/data
* Код метрики: https://www.kaggle.com/code/metric/hull-competition-sharpe
* Demo submission: https://www.kaggle.com/code/sohier/hull-tactical-market-prediction-demo-submission
* Отладка code-соревнований: https://www.kaggle.com/code-competition-debugging

Цитирование: Blair Hull, Petra Bakosova, Laurent Lanteigne, Aishvi Shah, Euan C Sinclair, Petri Fast, Will Raj, Harold Janecek, Sohier Dane, and Addison Howard. *Hull Tactical - Market Prediction*. https://kaggle.com/competitions/hull-tactical-market-prediction, 2025. Kaggle.
