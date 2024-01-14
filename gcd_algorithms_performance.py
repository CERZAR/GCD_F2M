import argparse
import datetime

import matplotlib.pyplot as plt

from common import DEGREE_FILE, AMOUNT_FILE


def read_data_from_file(filename):
    with open(filename, 'r') as file:
        return [eval(line.strip()) for line in file]


def filter_data(data, key):
    return [d for d in data if d[0] == key]


def extract_numeric_data(data, index_x, index_y):
    x = [int(d[index_x]) for d in data]
    y = [float(d[index_y]) for d in data]
    return x, y


def plot_graph(x, y, label, xlabel, ylabel, title):
    plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')


def main():
    parser = argparse.ArgumentParser(description='Отрисовка графиков результатов')
    parser.add_argument('--m', type=int, help='Степень поля F2^m')
    parser.add_argument('--degree', type=int, help='Максимальная степень (m) полиномов в поле F2^m')
    parser.add_argument('--amount', type=int, help='Количество полиномов в поле F2^m')
    args = parser.parse_args()

    if not args.degree or not args.amount:
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        name = f'graph{current_time}.png'
        amount = degree = m_value = None
    else:
        amount = args.amount
        degree = args.degree
        m_value = args.m
        name = f'{amount}_{degree}_{m_value}.png'

    # Чтение данных из файлов
    amount_data = read_data_from_file(AMOUNT_FILE)
    degrees_data = read_data_from_file(DEGREE_FILE)

    # Разделение данных по реализации (параллельная и последовательная)
    p_amount_data = filter_data(amount_data, 'p')
    s_amount_data = filter_data(amount_data, 's')
    p_degrees_data = filter_data(degrees_data, 'p')
    s_degrees_data = filter_data(degrees_data, 's')

    # Преобразование данных в числовой формат
    p_count_x, p_count_y = extract_numeric_data(p_amount_data, 2, 3)
    s_count_x, s_count_y = extract_numeric_data(s_amount_data, 2, 3)
    p_degrees_x, p_degrees_y = extract_numeric_data(p_degrees_data, 1, 3)
    s_degrees_x, s_degrees_y = extract_numeric_data(s_degrees_data, 1, 3)

    # Построение графиков
    plt.figure(figsize=(15, 6))

    # График зависимости времени работы от количества полиномов
    plt.subplot(1, 2, 1)
    amount_title = (f'Зависимость времени работы от количества полиномов (степеней: {degree}, m={m_value})' if degree
                    else 'Зависимость времени работы от количества полиномов')
    plot_graph(p_count_x, p_count_y, 'Параллельная', 'Количество полиномов', 'Время работы (сек)', amount_title)
    plot_graph(s_count_x, s_count_y, 'Последовательная', 'Количество полиномов', 'Время работы (сек)', amount_title)

    # График зависимости времени работы от степени полинома
    plt.subplot(1, 2, 2)
    degree_title = (f'Зависимость времени работы от степени полинома (полиномов: {amount}, m={m_value})' if amount
                    else 'Зависимость времени работы от степени полинома')
    plot_graph(p_degrees_x, p_degrees_y, 'Параллельная', 'Степень полинома', 'Время работы (сек)', degree_title)
    plot_graph(s_degrees_x, s_degrees_y, 'Последовательная', 'Степень полинома', 'Время работы (сек)', degree_title)

    plt.tight_layout()
    plt.savefig(f'results/graph{name}.png')  # Save the first subplot


if __name__ == '__main__':
    main()
