import ijson
import json


def extract_texts_with_ids(file_path, output_file):
    """
    Извлекает все значения полей 'text' из большого JSON, добавляет идентификаторы и сохраняет их в новый файл.
    Пропускает пустые строки.
    """
    texts = []
    total_items = count_items(file_path)
    processed_count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        parser = ijson.items(file, 'item')
        for item in parser:
            for text in find_texts(item):
                if text.strip():
                    texts.append({'id': len(texts) + 1, 'text': text})

            processed_count += 1
            if processed_count % 1000 == 0 or processed_count == total_items:
                completion = (processed_count / total_items) * 100
                print(f"\rПрогресс: {completion:.2f}% ({processed_count} из {total_items})", end="")

    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(texts, out_file, ensure_ascii=False, indent=4)

    print(f"\nНайдено {len(texts)} текстов. Результат сохранён в {output_file}.")


def count_items(file_path):
    """
    Считает количество элементов верхнего уровня в JSON.
    """
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        parser = ijson.items(file, 'item')
        for _ in parser:
            count += 1
    return count


def find_texts(data):
    """
    Рекурсивно ищет все поля 'text' в структуре данных.
    """
    found_texts = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'text' and isinstance(value, str):
                found_texts.append(value)
            elif isinstance(value, (dict, list)):
                found_texts.extend(find_texts(value))

    elif isinstance(data, list):
        for item in data:
            found_texts.extend(find_texts(item))

    return found_texts


def main():
    file_path = 'walls.json'
    output_file = 'texts_with_ids.json'

    print("Считаем общее количество элементов...")
    total_items = count_items(file_path)
    print(f"Общее количество элементов: {total_items}")

    print("Извлекаем тексты с идентификаторами...")
    extract_texts_with_ids(file_path, output_file)


if __name__ == "__main__":
    main()
