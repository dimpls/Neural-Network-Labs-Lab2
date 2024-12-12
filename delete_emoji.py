import csv
import json
import re

emoji_dict = {
    "😀": "широкая улыбка",
    "😁": "улыбка с зубами",
    "😂": "смех",
    "🤣": "смех до слез",
    "😃": "радость",
    "😄": "радостная улыбка",
    "😅": "нервный смех",
    "😆": "смеющийся",
    "😉": "подмигивание",
    "😊": "улыбка",
    "😋": "вкусно",
    "😎": "в крутых очках",
    "😍": "восторг",
    "😘": "поцелуй",
    "😗": "поцелуй",
    "😙": "поцелуй с улыбкой",
    "😚": "поцелуй с закрытыми глазами",
    "🙂": "легкая улыбка",
    "🙃": "перевернутая улыбка",
    "😌": "облегчение",
    "🥰": "влюбленность",
    "🤩": "восхищение",
    "😇": "ангелочек",
    "🤔": "размышление",
    "🤨": "подозрение",
    "😐": "нейтральное лицо",
    "😑": "без эмоций",
    "😶": "молчание",
    "🙄": "скептический взгляд",
    "😏": "самодовольство",
    "😒": "раздражение",
    "😞": "разочарование",
    "😟": "обеспокоенность",
    "😤": "раздражение",
    "😢": "грусть",
    "😭": "слёзы",
    "😦": "шок",
    "😧": "испуг",
    "😨": "ужас",
    "😩": "усталость",
    "🤯": "взрыв мозга",
    "😬": "нервное выражение",
    "😱": "крик",
    "😳": "смущение",
    "🥵": "жара",
    "🥶": "холод",
    "😵": "головокружение",
    "🤢": "тошнота",
    "🤮": "рвота",
    "🤧": "чихание",
    "😇": "ангел",
    "💔": "разбитое сердце",
    "❤️": "любовь",
    "🧡": "оранжевое сердце",
    "💛": "желтое сердце",
    "💚": "зеленое сердце",
    "💙": "синее сердце",
    "💜": "фиолетовое сердце",
    "🖤": "черное сердце",
    "🤍": "белое сердце",
    "💕": "двойная любовь",
    "💞": "вращающиеся сердца",
    "💓": "бьющееся сердце",
    "💗": "растущее сердце",
    "💖": "светящееся сердце",
    "💘": "сердце со стрелой",
    "💝": "подарочное сердце",
    "💟": "декоративное сердце",

    "🐰": "зайчик",
    "🐱": "кошка",
    "🐶": "собака",
    "🦊": "лиса",
    "🐻": "медведь",
    "🐼": "панда",
    "🐨": "коала",
    "🐯": "тигр",
    "🦁": "лев",
    "🐮": "корова",
    "🐷": "свинья",
    "🐸": "лягушка",
    "🐵": "обезьяна",
    "🐔": "курица",
    "🐧": "пингвин",
    "🐦": "птичка",
    "🐤": "цыпленок",
    "🐺": "волк",
    "🐗": "кабан",
    "🐴": "лошадь",
    "🦄": "единорог",
    "🐛": "гусеница",
    "🐝": "пчела",
    "🐞": "божья коровка",
    "🦋": "бабочка",
    "🐢": "черепаха",
    "🐍": "змея",
    "🐬": "дельфин",
    "🐳": "кит",
    "🐋": "синий кит",
    "🐙": "осьминог",
    "🐟": "рыба",
    "🦀": "краб",
    "🦐": "креветка",
    "🦑": "кальмар",

    "✨": "блеск",
    "🔥": "огонь",
    "💧": "капля",
    "☀️": "солнце",
    "🌞": "улыбающееся солнце",
    "🌚": "луна",
    "🌝": "улыбающаяся луна",
    "🌈": "радуга",
    "🍀": "клевер",
    "🌸": "цветок",
    "🌹": "роза",
    "🌷": "тюльпан",
    "🌺": "гибискус",
    "🌼": "ромашка",
    "💐": "букет",

    "🎉": "праздник",
    "🎊": "конфетти",
    "🎁": "подарок",
    "🎈": "воздушный шар",
    "🕯": "свеча",
    "⚽️": "футбольный мяч",
    "🏀": "баскетбольный мяч",
    "🎮": "игровой джойстик",
    "🎵": "нота",
    "🎶": "ноты",
    "📚": "книги",
    "🖊": "ручка",
    "📱": "телефон",
    "💻": "ноутбук",
    "🎥": "камера",

    "💃": "танец",
    "🕺": "танцующий",
    "🙏": "молитва",
    "👏": "аплодисменты",
    "👍": "лайк",
    "👎": "дизлайк",
    "👋": "приветствие",
    "🤝": "рукопожатие",
    "🤗": "объятия",
    "✌️": "победа",
    "🤟": "рок",
    "👌": "окей",
    "✋": "стоп",
    "🤚": "поднятая рука",
    "😼": "хитрая кошка",
    "🚬": "сигарета",
    "🤤": "слюни от удовольствия",
    "😔": "печаль",
    "😮": "удивление",
    "🥺": "умоляющий взгляд",
    "🤭": "прикрытая улыбка",
    "🤦🏼‍": "рука на лице",
    "☀️": "солнце",
    "🤙": "жест шака",
    "🌙": "луна",
    "👉": "показывает направо",
    "👈": "показывает налево",
    "‍♀": "женщина",
    "🤡": "клоун",
    "👊🏻": "удар кулаком",
    "🤬": "злость",
    "🤕": "больной",
    "😜": "веселье с подмигиванием",
    "💅": "ухоженные ногти",
    "🥀": "увядший цветок",
    "😠": "злость",
    "💏": "поцелуй пары",
"🕎": "менора, символ иудаизма",
    "杂": "иероглиф, означающий 'смешанный'",
    "⛎": "змееносец, знак зодиака",
    "📊": "диаграмма или график",
    "💩": "кучка, символ юмора или недовольства",
    "🎃": "тыква на Хэллоуин",
    "🌂": "зонтик",
    "🔑": "ключ, символ секрета или открытия",
    "🇷": "буква R в флаге",
    "𝗟": "латинская буква L в жирном стиле",
    "🧎": "на коленях, символ покаяния или просьбы",
    "금": "корейский иероглиф 'золото'",
    "尾": "хвост на китайском",
    "🌨": "снегопад",
    "🔭": "телескоп",
    "🤾": "человек, играющий в мяч",
    "𝙙": "латинская буква d в рукописном стиле",
    "🏺": "греческая амфора",
    "💰": "мешок с деньгами, символ богатства",
    "🍪": "печенье",
    "狗": "собака на китайском",
    "🔦": "фонарик",
    "🪄": "волшебная палочка",
    "🌿": "растение, символ природы",
    "🄲": "буква C в квадрате",
    "该": "китайский иероглиф, означающий 'должен'",
    "🫱": "рука в движении",
    "猴": "обезьяна на китайском",
    "𝕕": "латинская буква d в готическом стиле",
    "🚠": "канатная дорога",
    "。": "точка на японском",
    "🏝": "остров",
    "🏐": "волейбольный мяч",
    "🛢": "бочка с жидкостью",
    "🛡": "щит, символ защиты",
    "车": "машина на китайском",
    "𝔦": "латинская буква i в каллиграфическом стиле",
    "字": "символ 'буква' на китайском",
    "신": "новый на корейском",
    "💆": "человек, которому делают массаж",
    "➢": "стрелка",
    "🕷": "паук",
    "◌": "пустой круг",
    "😖": "страдание или замешательство",
    "🎴": "японская карточная игра",
    "♍": "дева, знак зодиака",
    "🕸": "паучья паутина",
    "タ": "японская буква 'та'",
    "岂": "вопросительное слово на китайском",
    "ｙ": "буква y в японском стиле",
    "🍧": "японское мороженое 'какигори'",
    "🎍": "новогоднее украшение в Японии",
    "𝗧": "латинская буква T в жирном стиле",
    "𝟕": "цифра 7",
    "𝐖": "латинская буква W в жирном стиле",
    "当": "китайский иероглиф, означающий 'быть'",
    "害": "китайский иероглиф, означающий 'вред'",
    "・": "японская межсловная точка",
    "💸": "деньги, символ трат",
    "🤑": "лицо с деньгами, символ богатства",
    "殖": "китайский иероглиф, означающий 'расти'",
    "➤": "стрелка вправо",
    "🎌": "японский национальный флаг",
    "㊙": "японский символ 'секрет'",
    "🈺": "открыто для бизнеса на японском",
    "⤵": "стрелка вниз",
    "▒": "узор с прямоугольниками",
    "🍯": "банка мёда",
    "🧟": "зомби",
    "💮": "белый цветок, символ чистоты",
    "来": "китайский иероглиф, означающий 'приходить'",
    "男": "мужчина на китайском",
    "𝙠": "латинская буква k в рукописном стиле",
    "🔖": "закладка",
    "🔫": "пистолет",
    "🔃": "кнопка обновления",
    "✏": "карандаш",
    "𝖸": "латинская буква Y в готическом стиле",
    "😕": "смущение",
    "ラ": "японская буква 'ра'",
    "Ｏ": "буква O в японском стиле",
    "🕞": "время: 3:30",
    "国": "страна на китайском",
    "🄻": "буква L в квадрате",
    "⛈": "грозовой дождь",
    "🪅": "мексиканская пиньята",
    "道": "путь на китайском",
    "⬇": "стрелка вниз",
    "👽": "инопланетянин",
    "💡": "лампочка, символ идеи",
    "🈵": "заполнено на японском",
    "，": "запятая на китайском",
    "🔂": "повтор одного",
    "🎄": "новогодняя ёлка",
    "🕜": "время: 1:30",
    "👇": "палец, указывающий вниз",
    "🖥": "компьютер",
    "🃏": "джокер",
    "他": "он на китайском",
    "🧖": "человек в сауне",
    "耻": "стыд на китайском",
    "艹": "трава (иероглиф)",
    "🕎": "менора, символ иудаизма",
    "杂": "иероглиф, означающий 'смешанный'",
    "⛎": "змееносец, знак зодиака",
    "📊": "диаграмма или график",
    "💩": "кучка, символ юмора или недовольства",
    "🎃": "тыква на Хэллоуин",
    "🌂": "зонтик",
    "🔑": "ключ, символ секрета или открытия",
    "🇷": "буква R в флаге",
    "𝗟": "латинская буква L в жирном стиле",
    "🧎": "на коленях, символ покаяния или просьбы",
    "금": "корейский иероглиф 'золото'",
    "尾": "хвост на китайском",
    "🌨": "снегопад",
    "🔭": "телескоп",
    "🤾": "человек, играющий в мяч",
    "𝙙": "латинская буква d в рукописном стиле",
    "🏺": "греческая амфора",
    "💰": "мешок с деньгами, символ богатства",
    "🍪": "печенье",
    "狗": "собака на китайском",
    "🔦": "фонарик",
    "🪄": "волшебная палочка",
    "🌿": "растение, символ природы",
    "🄲": "буква C в квадрате",
    "该": "китайский иероглиф, означающий 'должен'",
    "🫱": "рука в движении",
    "猴": "обезьяна на китайском",
    "𝕕": "латинская буква d в готическом стиле",
    "🚠": "канатная дорога",
    "。": "точка на японском",
    "🏝": "остров",
    "🏐": "волейбольный мяч",
    "🛢": "бочка с жидкостью",
    "🛡": "щит, символ защиты",
    "车": "машина на китайском",
    "𝔦": "латинская буква i в каллиграфическом стиле",
    "字": "символ 'буква' на китайском",
    "신": "новый на корейском",
    "💆": "человек, которому делают массаж",
    "➢": "стрелка",
    "🕷": "паук",
    "◌": "пустой круг",
    "😖": "страдание или замешательство",
    "🎴": "японская карточная игра",
    "♍": "дева, знак зодиака",
    "🕸": "паучья паутина",
    "タ": "японская буква 'та'",
    "岂": "вопросительное слово на китайском",
    "ｙ": "буква y в японском стиле",
    "🍧": "японское мороженое 'какигори'",
    "🎍": "новогоднее украшение в Японии",
    "𝗧": "латинская буква T в жирном стиле",
    "𝟕": "цифра 7",
    "𝐖": "латинская буква W в жирном стиле",
    "当": "китайский иероглиф, означающий 'быть'",
    "害": "китайский иероглиф, означающий 'вред'",
    "・": "японская межсловная точка",
    "💸": "деньги, символ трат",
    "🤑": "лицо с деньгами, символ богатства",
    "殖": "китайский иероглиф, означающий 'расти'",
    "➤": "стрелка вправо",
    "🎌": "японский национальный флаг",
    "㊙": "японский символ 'секрет'",
    "🈺": "открыто для бизнеса на японском",
    "⤵": "стрелка вниз",
    "▒": "узор с прямоугольниками",
    "🍯": "банка мёда",
    "🧟": "зомби",
    "💮": "белый цветок, символ чистоты",
    "来": "китайский иероглиф, означающий 'приходить'",
    "男": "мужчина на китайском",
    "𝙠": "латинская буква k в рукописном стиле",
    "🔖": "закладка",
    "🔫": "пистолет",
    "🔃": "кнопка обновления",
    "✏": "карандаш",
    "𝖸": "латинская буква Y в готическом стиле",
    "😕": "смущение",
    "ラ": "японская буква 'ра'",
    "Ｏ": "буква O в японском стиле",
    "🕞": "время: 3:30",
    "国": "страна на китайском",
    "🄻": "буква L в квадрате",
    "⛈": "грозовой дождь",
    "🪅": "мексиканская пиньята",
    "道": "путь на китайском",
    "⬇": "стрелка вниз",
    "👽": "инопланетянин",
    "💡": "лампочка, символ идеи",
    "🈵": "заполнено на японском",
    "，": "запятая на китайском",
    "🔂": "повтор одного",
    "🎄": "новогодняя ёлка",
    "🕜": "время: 1:30",
    "👇": "палец, указывающий вниз",
    "🖥": "компьютер",
    "🃏": "джокер",
    "他": "он на китайском",
    "🧖": "человек в сауне",
    "耻": "стыд на китайском",
    "艹": "трава (иероглиф)",
"👊": "кулак, символ удара или солидарности",
    "🍎": "яблоко, символ здоровья или знаний",
    "🍉": "арбуз, символ лета или свежести",
    "🍇": "виноград, символ изобилия",
    "🍌": "банан, символ тропиков или юмора",
    "🍋": "лимон, символ кислинки или свежести",
    "🍍": "ананас, символ тропического рая",
    "🍒": "вишня, символ сладости или романтики",
    "🍓": "клубника, символ лета или романтики",
    "🍈": "дыня, символ свежести и лета",
    "🥭": "манго, символ тропиков или сладости",
    "🍑": "персик, иногда используется как символ части тела",
    "🍐": "груша, символ свежести",
    "🍏": "зелёное яблоко, символ здоровья",
    "🍋": "лимон, символ кислого вкуса",
    "🍔": "гамбургер, символ еды",
    "🍟": "картофель фри, символ быстрого питания",
    "🍕": "пицца, символ итальянской кухни",
    "🌮": "тако, символ мексиканской кухни",
    "🌭": "хот-дог, символ уличной еды",
    "🍿": "попкорн, символ кино или развлечений",
    "🥞": "блинчики, символ завтрака",
    "🥯": "бейгл, символ выпечки",
    "🍗": "куриный окорочок, символ еды",
    "🍖": "мясо на кости, символ еды",
    "🍳": "сковорода с яичницей, символ завтрака",
    "🥘": "кастрюля с едой, символ готовки",
    "🍝": "паста, символ итальянской кухни",
    "🥗": "салат, символ здоровья",
    "🍲": "суп, символ домашнего уюта",
    "🥣": "миска с едой, символ завтрака или супа",
    "🍤": "креветка, символ морепродуктов",
    "🍦": "мороженое, символ лета и сладости",
    "🍨": "чашка мороженого, символ удовольствия",
    "🍧": "японское мороженое 'какигори'",
    "🎂": "торт, символ праздника или дня рождения",
    "🍰": "кусочек торта, символ десерта",
    "🧁": "капкейк, символ маленького удовольствия",
    "🥧": "пирог, символ выпечки",
    "🍪": "печенье, символ сладости",
    "🍩": "пончик, символ сладости и расслабления",
    "🍫": "шоколад, символ сладости и любви",
    "🍬": "конфета, символ радости",
    "🍭": "леденец, символ сладкого удовольствия",
    "🍮": "карамельный пудинг, символ десерта",
    "🍯": "мёд, символ сладости и природы",
    "🍼": "бутылочка с молоком, символ детства",
    "☕": "чашка кофе, символ бодрости",
    "🍵": "чашка зелёного чая, символ спокойствия",
    "🍶": "японская бутылка саке, символ японской культуры",
    "🍹": "коктейль, символ праздника или отдыха",
    "🍸": "бокал для мартини, символ вечеринки",
    "🍷": "бокал вина, символ релаксации",
    "🥂": "чокающиеся бокалы, символ праздника",
    "🍻": "чокающиеся кружки пива, символ веселья",
    "🥃": "стакан виски, символ крепкого напитка",
    "🥤": "напиток с трубочкой, символ прохлады",
    "🧋": "бабл-чай, символ современной культуры",
    "🍽": "вилка и нож, символ еды",
    "🍴": "вилка и ложка, символ еды",
    "🥢": "палочки для еды, символ азиатской кухни",
    "🥄": "ложка, символ еды или готовки",
    "🔪": "нож, символ готовки",
    "⚱": "урна, символ памяти",
    "💣": "бомба, символ угрозы или неожиданности",
    "💡": "лампочка, символ идеи",
    "🕯": "свеча, символ уюта или памяти",
    "🔦": "фонарик, символ света",
    "📜": "свиток, символ старины или знаний",
    "📃": "бумажный документ, символ текста",
    "📄": "лист бумаги, символ работы",
    "📋": "блокнот с зажимом, символ записей",
    "📌": "канцелярская кнопка, символ закрепления",
    "📎": "скрепка, символ офиса",
    "📏": "линейка, символ измерения",
    "📐": "угольник, символ точности",
    "📅": "календарь, символ даты",
    "📆": "настенный календарь, символ планирования",
    "🗓": "ежедневник, символ расписания",
    "📖": "открытая книга, символ чтения",
    "📗": "зелёная книга, символ знаний",
    "📕": "красная книга, символ запрета",
    "📘": "синяя книга, символ информации",
    "📙": "оранжевая книга, символ обучения",
    "📚": "куча книг, символ учёбы",
    "📓": "тетрадь, символ записей",
    "📔": "тетрадь с закладкой, символ важных записей",
    "📒": "жёлтая тетрадь, символ записей",
    "📜": "документ, символ свитка",
    "📑": "разделённый документ, символ упорядоченности",
    "📝": "лист с текстом, символ написания",
    "✏": "карандаш, символ письма",
    "✒": "ручка-перо, символ классического письма",
    "🖋": "ручка, символ работы",
    "🖌": "кисть, символ рисования",
    "🖍": "цветной мелок, символ творчества",
    "🗂": "папка с разделителями, символ офиса",
    "📁": "открытая папка, символ файлов",
    "📂": "закрытая папка, символ архива",
    "🗄": "шкаф с файлами, символ офиса",
    "📦": "коробка, символ посылки",
    "🛒": "тележка для покупок, символ шопинга",
    "🛍": "сумка с покупками, символ торговли",
    "🎁": "подарочная коробка, символ праздника",
"📮": "почтовый ящик для отправки писем",
    "📬": "почтовый ящик с входящими письмами",
    "📭": "почтовый ящик с отправленными письмами",
    "📯": "почтовый рожок, символ оповещения",
    "🗳": "урна для голосования",
    "📤": "ящик с отправкой, символ исходящих писем",
    "📥": "ящик с входящими, символ входящих писем",
    "📦": "коробка, символ доставки или посылки",
    "📫": "закрытый почтовый ящик, символ почты",
    "📪": "открытый почтовый ящик, символ почты",
    "🖊": "ручка, символ записей или подписей",
    "🖍": "цветной мелок, символ творчества",
    "📋": "блокнот с зажимом, символ записей",
    "🗒": "лист бумаги, символ заметок",
    "🗓": "ежедневник, символ календаря",
    "📅": "календарь, символ времени и даты",
    "📆": "настенный календарь, символ планирования",
    "📌": "канцелярская кнопка, символ закрепления информации",
    "📎": "скрепка, символ прикрепления файлов",
    "📏": "линейка, символ измерения",
    "📐": "угольник, символ точности",
    "📚": "куча книг, символ учёбы",
    "📖": "открытая книга, символ чтения",
    "📗": "зелёная книга, символ знаний или экологии",
    "📘": "синяя книга, символ научной литературы",
    "📙": "оранжевая книга, символ творчества",
    "📕": "красная книга, символ важности",
    "📓": "тетрадь, символ записей",
    "📔": "тетрадь с закладкой, символ учебных заметок",
    "📒": "жёлтая тетрадь, символ канцелярии",
    "📜": "свиток, символ исторических документов",
    "📑": "разделённый документ, символ организации",
    "📝": "лист с текстом, символ письма или заметок",
    "✏": "карандаш, символ творчества или записей",
    "✒": "ручка-перо, символ каллиграфии",
    "📂": "открытая папка, символ файлов",
    "🗂": "папка с разделителями, символ упорядоченности",
    "📁": "закрытая папка, символ архива",
    "🗄": "шкаф с файлами, символ офиса",
    "📊": "диаграмма, символ данных или статистики",
    "📉": "график с понижением, символ падения",
    "📈": "график с ростом, символ успеха",
    "📋": "записной блокнот, символ деловых задач",
    "📇": "каталог карточек, символ файлов",
    "🗒": "заметочник, символ планирования",
    "🔖": "закладка, символ сохранения информации",
    "📑": "документ с вкладками, символ работы",
    "🖇": "двойная скрепка, символ канцелярии",
    "📜": "свиток, символ старины или знаний",
    "📄": "лист бумаги, символ текстовой информации",
    "📃": "официальный документ, символ важных данных",
    "📤": "ящик для исходящих, символ отправки информации",
    "📥": "ящик для входящих, символ получения информации",
    "🗳": "избирательная урна, символ голосования",
    "🔗": "ссылка, символ соединения или сети",
    "📦": "коробка, символ посылок",
    "🛒": "тележка для покупок, символ торговли",
    "🛍": "сумка с покупками, символ шопинга",
    "🎁": "подарок, символ праздника",
    "🧧": "красный конверт, символ удачи и китайского Нового года",
    "🎀": "бантик, символ украшения",
    "🎗": "лента, символ поддержки или памяти",
    "🎭": "театральная маска, символ искусства или представления",
    "🎨": "палитра, символ творчества или рисования",
    "🎤": "микрофон, символ музыки или выступления",
    "🎧": "наушники, символ музыки или звука",
    "🎼": "нотный стан, символ музыки",
    "🎹": "пианино, символ музыки",
    "🎷": "саксофон, символ джаза",
    "🎸": "гитара, символ музыки",
    "🎻": "скрипка, символ классической музыки",
    "🎼": "нотный стан, символ творчества",
    "🥁": "барабан, символ ритма или музыки",
    "🎺": "труба, символ оркестра",
    "🎚": "звуковой микшер, символ настройки звука",
    "🎛": "звукорежиссёрский пульт, символ музыки",
    "🎟": "билет, символ мероприятия",
    "🎫": "билет, символ входа на событие",
    "🎪": "цирк, символ веселья",
    "🎬": "хлопушка, символ киноиндустрии",
    "🎥": "видеокамера, символ съёмки",
    "📽": "киноаппарат, символ старинного кино",
    "📹": "видеокамера, символ записи",
    "📼": "видеокассета, символ старины",
    "🎞": "киноплёнка, символ фильма",
    "📸": "фотоаппарат, символ фотографии",
    "📷": "камера, символ фотосъёмки",
    "📺": "телевизор, символ развлечений",
    "📡": "спутниковая антенна, символ связи",
    "📻": "радиоприёмник, символ аудиосвязи",
    "🎙": "студийный микрофон, символ звукозаписи",
    "🎚": "регуляторы, символ настройки звука",
    "🎛": "звукорежиссёрский пульт, символ музыки",
    "🎷": "саксофон, символ музыки",
    "🎸": "гитара, символ музыки",
    "🎹": "пианино, символ музыки",
    "🥁": "барабан, символ ритма или музыки",
    "🎺": "труба, символ оркестра",
    "🎻": "скрипка, символ классической музыки",
    "🎼": "нотный стан, символ творчества",
    "🔜": "Стрелка вправо",
    "𝟩": "Цифра 5 (через специальный шрифт)",
    "⚾": "Бейсбольный мяч",
    "🍛": "Кари",
    "🆕": "Новый",
    "🍂": "Листопад",
    "🆄": "Символ для 'U'",
    "😺": "Кошка с улыбающимся лицом",
    "📿": "Четки",
    "𝖠": "Буква A (через специальный шрифт)",
    "🤷": "Пожимание плечами",
    "✍": "Написание",
    "看": "Смотреть (китайский символ)",
    "🥬": "Листовая капуста",
    "🐘": "Слон",
    "🥫": "Консервированная еда",
    "🗣": "Говорящий человек",
    "📩": "Конверт с письмом",
    "✓": "П галочка",
    "🗝": "Ключ",
    "인": "Человек (корейский символ)",
    "🚭": "Запрещено курить",
    "➔": "Стрелка вправо",
    "✌": "Знак мира",
    "❇": "Звезда",
    "❹": "Цифра 4 (круглая)",
    "𝗲": "Буква e (через специальный шрифт)",
    "🅄": "Символ U в круге",
    "👤": "Человек",
    "✈": "Самолет",
    "🍡": "Шашлычки",
    "🧝": "Эльф",
    "🙈": "Обезьяна, закрывающая глаза",
    "－": "Дефис",
    "养": "Воспитание (китайский символ)",
    "😝": "Лицо с языком наружу",
    "意": "Мысль (китайский символ)",
    "东": "Восток (китайский символ)",
    "🔣": "Символы",
    "🚹": "Мужской туалет",
    "种": "Сорт (китайский символ)",
    "要": "Нужно (китайский символ)",
    "丫": "Ноги (китайский символ)",
    "𝐍": "Буква N (через специальный шрифт)",
    "干": "Сушить, делать (китайский символ)",
    "◼": "Черный квадрат",
    "🚕": "Такси",
    "🧯": "Огнетушитель",
    "🧘": "Человек, занимающийся йогой",
    "⛰": "Гора",
    "🐜": "Муравей",
    "𝗮": "Буква a (через специальный шрифт)",
    "＋": "Плюс",
    "💯": "Сто процентов",
    "🕠": "5:30",
    "🅒": "Символ C в круге",
    "🤌": "Пальцы на руке",
    "▂": "Линия",
    "ご": "Го (японский символ)",
    "🌡": "Термометр",
    "👯": "Два человека с вуалью",
    "🥓": "Бекон",
    "🔁": "Повтор",
    "臭": "Пахнет (китайский символ)",
    "在": "Находиться (китайский символ)",
    "𝙧": "Буква r (через специальный шрифт)",
    "🔱": "Трезубец",
    "⚙": "Шестеренка",
    "👩": "Женщина",
    "𝐕": "Буква V (через специальный шрифт)",
    "𝘾": "Буква C (через специальный шрифт)",
    "𝓝": "Буква N (через специальный шрифт)",
    "🩸": "Капля крови",
    "지": "Земля (корейский символ)",
    "唔": "Не (китайский символ)",
    "🎿": "Лыжи",
    "🎲": "Кубик",
    "🈯": "Отмечено (японский символ)",
    "🏋": "Человек, поднимающий штангу",
    "🇨": "Флаг Китая",
    "❘": "Точка",
    "🏖": "Пляж",
    "𝙣": "Буква n (через специальный шрифт)",
    "┗": "Часть рамки",
    "🦆": "Утка",
    "🦼": "Ехидна",
    "𝟬": "Цифра 0",
    "多": "Много (китайский символ)",
    "唱": "Петь (китайский символ)",
    "🧾": "Документ",
    "𝙤": "Буква o (через специальный шрифт)",
    "🇦": "Флаг Алжира",
    "⤴": "Стрелка вверх",
    "🕕": "6:00",
    "👾": "Пришелец",
    "？": "Вопросительный знак",
    "同": "тот же, одинаковый",
    "🪖": "шлем",
    "🐖": "свинья",
    "😈": "злой смайлик",
    "𝟎": "ноль",
    "🐈": "кошка",
    "✅": "галочка",
    "🤹": "жонглёр",
    "☑": "галочка в квадрате",
    "𝕝": "латинская буква l с декоративным шрифтом",
    "𝟛": "три с декоративным шрифтом",
    "👆": "пальцы вверх",
    "𝖡": "латинская буква B с декоративным шрифтом",
    "╾": "горизонтальная линия",
    "🌰": "орех",
    "🏳": "белый флаг",
    "😿": "плачущая кошка",
    "🕌": "мечеть",
    "🇪": "флаг Испании",
    "🅔": "квадрат с буквой E",
    "⛽": "заправочная станция",
    "🤱": "кормящая мать",
    "𝕍": "латинская буква V с декоративным шрифтом",
    "🧗": "скалолаз",
    "🉐": "символ для разрешённого",
    "🧭": "компас",
    "こ": "японский символ для слова 'который'",
    "𝒂": "латинская буква a с декоративным шрифтом",
    "☃": "снеговик",
    "🚰": "фонтан с водой",
    "🍁": "кленовый лист",
    "🛎": "колокольчик",
    "🐥": "птенец",
    "🙍": "человек с поднятой рукой",
    "🈳": "символ для свободного",
    "太": "китайский символ для 'очень'",
    "😓": "потящийся смайлик",
    "📠": "факс",
    "👟": "кроссовки",
    "⛑": "спасательный шлем",
    "⛔": "запрещено",
    "💽": "дискета",
    "🧷": "иголка с ниткой",
    "🅼": "квадрат с буквой M",
    "得": "китайский символ для 'достигать'",
    "♬": "музыкальные ноты",
    "🚤": "моторная лодка",
    "🫡": "отдающий честь",
    "🔇": "без звука",
    "🪃": "сундук",
    "入": "входить",
    "🅾": "квадрат с буквой O",
    "♟": "шахматная фигура - пешка",
    "🏔": "горный пик",
    "𝐬": "латинская буква s с декоративным шрифтом",
    "♿": "символ доступности",
    "👨": "мужчина",
    "게": "корейский символ для слова 'игра'",
    "𝕪": "латинская буква y с декоративным шрифтом",
    "二": "китайский символ для 'два'",
    "脸": "китайский символ для 'лицо'",
    "キ": "японский символ для 'ки'",
    "✂": "ножницы",
    "🕴": "человек в костюме",
    "𝕒": "латинская буква a с декоративным шрифтом",
    "니": "корейский символ для 'ты'",
    "🤠": "ковбойская шляпа",
    "爪": "когти",
    "人": "человек",
    "🚚": "грузовик",
    "🫢": "удивлённый человек с прикрытым лицом",
    "✝": "крест",
    "🚙": "пикап",
    "🧺": "корзина",
    "𝟢": "ноль с декоративным шрифтом",
    "🐚": "ракушка",
    "🪒": "бритва",
    "♈": "овен (зодиакальный знак)",
    "差": "китайский символ для 'разница'",
    "🪡": "игла для шитья",
    "🅿": "квадрат с буквой P",
    "🚁": "вертолет",
    "Ｙ": "латинская буква Y с декоративным шрифтом",
    "𝗗": "латинская буква D с декоративным шрифтом",
    "⬈": "стрелка вверх и вправо",
    "🐾": "следы животных",
    "喔": "китайский символ для выражения удивления",
    "厌": "китайский символ для 'недовольство'",
    "🕣": "время 15:00",
    "也": "китайский символ для 'тоже'",
    "🙅": "отказ",
    "🏻": "кожа светлая",
    "错": "китайский символ для 'ошибка'",
    "🇩": "флаг Германии",
    "🇻": "флаг Венгрии",
    "🗞": "газета",
    "翘": "китайский символ для 'восходящий'",
    "🥚": "яйцо",
    "🥿": "женская туфля",
    "🥝": "киви",
    "🛺": "рикша",
    "🏨": "отель",
    "📣": "динамик",
    "👡": "туфли на каблуке",
    "𝓔": "латинская буква E с декоративным шрифтом",
    "🧒": "ребёнок",
    "真": "китайский символ для 'истинный'",
    "⛟": "символ транспорта",
    "畜": "китайский символ для 'домашние животные'",
    "🫰": "пальцы в знак любви",
    "🅴": "квадрат с буквой E",
    "🦎": "ящерица",
    "🍃": "листья",
    "子": "китайский символ для 'ребёнок'",
    "𝙞": "латинская буква i с декоративным шрифтом",
    "☁": "облако",
    "🇹": "флаг Таиланда",
    "🦃": "индюк",
    "⠀": "пробел",
    "🏂": "сноубордист",
    "👰": "невеста",
    "🗻": "гора Фудзи",
    "𝙖": "латинская буква a с декоративным шрифтом",
    "🫘": "бобы",
    "🏊": "плавец",
    "﻿": "неразрывный пробел",
    "🐆": "пантера",
    "𝓿": "латинская буква v с декоративным шрифтом",
    "🚈": "поезд",
    "💿": "компакт-диск",
    "🌛": "полумесяц",
    "🕊": "голубь",
    "🎩": "цилиндр",
    "🔳": "квадрат с диагональными линиями",
    "𝐯": "латинская буква v с декоративным шрифтом",
    "☝": "пальцем вверх",
    "🛴": "сигвей",
    "▓": "массив",
    "♨": "горячие источники",
    "𝟏": "один с декоративным шрифтом",
    "直": "китайский символ для 'прямой'",
    "🦉": "сова",
    "이": "корейский символ для 'это'",
    "🐠": "рыба",
    "𝙇": "латинская буква L с декоративным шрифтом",
    "리": "корейский символ для 'ре'",
    "大": "китайский символ для 'большой'",
    "𝗹": "латинская буква l с декоративным шрифтом",
    "臊": "китайский символ для 'резкий запах'",
    "🆔": "идентификатор",
    "🕧": "время 15:30",
    "弯": "китайский символ для 'кривой'",
    "𝟙": "единица с декоративным шрифтом",
    "♑": "козерог (зодиакальный знак)",
    "✉": "конверт",
    "🎮": "игровой контроллер",
    "🏛": "здание парламента",
    "🪤": "медвежий капкан",
    "🪗": "аккордеон",
    "💉": "шприц",
    "🛁": "ванна",
    "🐘": "слон",
    "🐺": "волк",
    "👂": "человек, указывающий пальцем",
    "🍜": "рамен",
    "🍋": "лимон",
    "🧸": "медвежонок",
    "🎻": "скрипка",
    "🌷": "тюльпан",
    "🚀": "ракета",
    "⛅": "облачно с прояснениями",
    "🔌": "штепсельная вилка",
    "📱": "мобильный телефон",
    "🚏": "остановка",
    "🪑": "стул",
    "🎰": "игровой автомат",
    "🛶": "каноэ",
    "🌐": "глобус с сеткой",
    "🖥": "настольный компьютер",
    "🛒": "тележка для покупок",
    "📚": "книги",
    "🛸": "летающая тарелка",
    "🖋": "перо",
    "🦅": "орел",
    "⛸": "коньки",
    "🔒": "замок",
    "📦": "коробка",
    "⛎": "зодиакальный знак змеи",
    "💌": "письмо с сердцем",
    "🏓": "настольный теннис",
    "🌼": "цветок ромашки",
    "🎂": "торт",
    "📋": "блокнот",
    "🧳": "чемодан",
    "🕶": "очки",
    "🏠": "дом",
    "🍰": "кекс",
    "🎉": "конфетти",
    "🎁": "подарок",
    "📞": "телефонный аппарат",
    "🍎": "яблоко",
    "🚗": "автомобиль",
    "🎲": "кубик",
    "🪑": "стул",
    "⛴": "плавучий объект",
    "🪙": "монета",
    "🔧": "ключ",
    "🖱": "мышь компьютерная",
    "🍷": "вино",
    "🚉": "ж/д станция",
    "🍓": "клубника",
    "🍇": "виноград",
    "🏹": "лук и стрела",
    "🛷": "сани",
    "🎬": "кинохроника",
    "📯": "горн",
    "🎼": "музыкальные ноты",
    "🐿": "белка",
    "🚣": "гребля",
    "🐕": "собака",
    "📏": "линейка",
    "🌶": "перец чили",
    "🌟": "звезда",
    "🏕": "кемпинг",
    "🏑": "хоккейная клюшка и шайба",
    "🍷": "вино",
    "🏰": "замок",
    "🧗": "альпинист",
    "🏺": "сосуд",
    "🕹": "джойстик",
    "🧃": "сок",
    "🏠": "дом",
    "🛍": "сумка для покупок",
    "🎤": "микрофон",
    "🚁": "вертолет",
    "🔑": "ключ",
    "🧴": "спрей",
    "🖤": "черное сердце",
    "🔗": "ссылка",
    "🎯": "мишень",
    "🎳": "боулинг",
    "🚢": "корабль",
    "🎮": "игровая приставка",
    "🔨": "молоток",
    "📷": "фотоаппарат",
    "🖲": "трекбол",
    "🍽": "столовая посуда",
    "⛴": "корабельная чайка",
    "🎻": "скрипка",
    "🏆": "трофей",
    "🎼": "музыкальная нота",
    "🎻": "скрипка"
}


def replace_emoji_with_text(text):
    for emoji, description in emoji_dict.items():
        text = text.replace(emoji, f" {description} ")
    return text


input_file = "processed_data/train_dataset_final.csv"
output_file = "train_dataset_final_v2.csv" 

with open(input_file, mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    headers = next(reader)  
    data = list(reader)

csv_data = [headers]  
for row in data:
    text = row[0]  
    label = row[1]  
    text = replace_emoji_with_text(text)
    csv_data.append([text, label])  

with open(output_file, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"CSV файл успешно создан: {output_file}")