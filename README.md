# ML

# Проекты модулей Data Analysis.

В данной папке присутствуют следующие проекты.

| Название проекта | Описание | Используемые библиотеки | 
| :---------------------- | :---------------------- | :---------------------- |
| Анализ Надежности Заёмщиков | Входные данные от кредитного отдела банка  — статистика о платёжеспособности клиентов. Очищены данные от выбросов, пропусков и дубликатов, а также преобразованы разные форматы данных. Заменены типы данных на соответствующие хранящимся данным. Удалены дубликаты. Выделены леммы в значениях столбца и категоризированны данные.Определена доля кредитоспособных клиентов.Проанализировано влияние семейного положения и количества детей клиента на факт возврата кредита в срок. Построена модель кредитного скоринга — специальной системы, которая оценивает способность потенциального заёмщика вернуть кредит банку.| *pandas* *pymystem3* |
| Анализ Объявлений Недвижимости | Проведен исследовательский анализ и предобработка данных для датасета с объявлениями о продаже квартир в Санкт-Петербурге. Выявлены, влияние площади, потолков, количества комнат, даты объявления на цены квартир всех представленных населённых пунктов и центра Санкт-Петербурга для построения автоматизированной системы определения цен во избежание мошенничества и аномалий. На основе данных сервиса Яндекс.Недвижимость определена рыночная стоимость объектов недвижимости разного типа, типичные параметры квартир, в зависимости от удаленности от центра. Проведена предобработка данных. Добавлены новые данные. Построены гистограммы, боксплоты, диаграммы рассеивания. | *pandas* *matplotlib.pyplot* |
| Анализ Тарифов | Оператор мобильной связи выяснил: многие клиенты пользуются архивными тарифами. Проведен предварительный анализ использования тарифов на выборке клиентов, проанализировано поведение клиентов при использовании услуг оператора и рекомендованы оптимальные наборы услуг для пользователей. Проверены гипотезы о различии выручки абонентов разных тарифов и различии выручки абонентов из Москвы и других регионов. Определен выгодный тарифный план для корректировки рекламного бюджета. Разработана система, способная проанализировать поведение клиентов и предложить пользователям новый тариф. Построена модель для задачи классификации, которая выберет подходящий тариф. Построена модель с максимально большим значением accuracy. Доля правильных ответов доведена до 0.75. Проверены accuracy на тестовой выборке. | *pandas* *matplotlib.pyplot* *numpy* *scipy.stats*|
| Анализ Компьютерных Игр | Интернет-магазин продаёт по всему миру компьютерные игры. Из открытых источников доступны исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы. Выявлены параметры, определяющие успешность игры в разных регионах мира. Выявлен потенциально популярный продукт и спланированы рекламные кампании. Выбран актуальный период для анализа. Составлены портреты пользователей каждого региона. Проверены гипотезы: средние пользовательские рейтинги платформ Xbox One и PC одинаковые; средние пользовательские рейтинги жанров Action и Sports разные. При анализе использовались критерий Стьюдента для независимых выборок. | *pandas* *matplotlib.pyplot* *numpy* *scipy.stats* |
| Анализ Авиалиний SQL | Для российской авиакомпании, выполняющей внутренние пассажирские перевозки важно понять предпочтения пользователей, покупающих билеты на разные направления. Извлечены данные запросами на языке SQL и методами библиотеки PySpark. Изучена база данных и проанализирован спрос пассажиров на рейсы в города, где проходят крупнейшие культурные фестивали. Проверена гипотеза о различии среднего спроса на билеты во время проведения различных фестивалей и в обычное время. | *pandas* *matplotlib.pyplot* *numpy* *scipy.stats* |
