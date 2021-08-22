import pandas as pd
import matplotlib.pyplot as plt


def ex1():
    profit_list = df['total_profit'].values
    plt.figure('figuritatest')
    plt.plot(months, profit_list, label='Month-wise profit data of last year')
    plt.xlabel('month number')
    plt.ylabel('total profit')
    plt.title('Company profit per month')
    plt.xticks(months)
    plt.yticks([100e3, 200e3, 300e3, 400e3, 500e3])
    plt.grid()
    plt.show()


def ex2():
    profit_list = df['total_profit'].values
    plt.figure('figuritatest')
    plt.plot(months, profit_list, label='profit data of last year', color='r', marker='o', markerfacecolor='k',
             linestyle='-', linewidth=3)
    plt.xlabel('month number')
    plt.ylabel('total profit')
    plt.title('Company profit per month')
    plt.xticks(months)
    plt.yticks([100e3, 200e3, 300e3, 400e3, 500e3])
    plt.grid()
    plt.show()


def ex3():
    facecream_d = df['facecream'].values
    facewash_d = df['facewash'].values
    toothpaste_d = df['toothpaste'].values
    bathingsoap_d = df['bathingsoap'].values
    shampoo_d = df['shampoo'].values
    moisturizer_d = df['moisturizer'].values

    plt.figure()
    plt.plot(months, facecream_d, label='facecream sales data', marker='o', linewidth=3)
    plt.plot(months, facewash_d, label='facecream sales data', marker='o', linewidth=3)
    plt.plot(months, toothpaste_d, label='toothpaste sales data', marker='o', linewidth=3)
    plt.plot(months, bathingsoap_d, label='bathing soap sales data', marker='o', linewidth=3)
    plt.plot(months, shampoo_d, label='shampoo sales data', marker='o', linewidth=3)
    plt.plot(months, moisturizer_d, label='moisturizer sales data', marker='o', linewidth=3)
    plt.xlabel('month number')
    plt.ylabel('sold units')
    plt.legend(loc='upper left')
    plt.xticks(months)
    plt.yticks([1e3, 2e3, 4e3, 6e3, 8e3, 10e3, 12e3, 15e3, 18e3])
    plt.grid()
    plt.title('sales data')
    plt.show()


def ex4():
    toothpaste_d = df['toothpaste'].values
    plt.figure()
    plt.scatter(months, toothpaste_d, label='toothpaste sale data')
    plt.xlabel('months')
    plt.ylabel('toothpaste sold units')
    plt.legend(loc='upper left')
    plt.grid(True, linewidth=0.5, linestyle='-')
    plt.show()


def ex5():
    bathingsoap_d = df['bathingsoap'].values
    plt.figure()
    plt.bar(months, bathingsoap_d)
    plt.savefig('sale_d_bathingsoap.png', dpi=150)
    plt.show()


def ex6():
    profit_list = df['total_profit'].values
    plt.figure()
    profit_range = [150e3, 175e3, 200e3, 225e3, 250e3, 300e3, 350e3]
    plt.hist(profit_list, profit_range, label='Profit data')
    plt.xticks(profit_range)
    plt.show()


def ex7():
    bathsoap = df['bathingsoap'].values
    facewash = df['facewash'].values
    f, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(months, facewash, label='facewash sale data', color='k', marker='o', linewidth=3)
    axs[0].grid()
    axs[1].plot(months, bathsoap, label='bathsoap sale data', color='r', marker='o', linewidth=3)
    axs[1].grid()
    plt.xticks(months)
    plt.xlabel('MONTH N')
    plt.ylabel('Sales units')
    plt.show()


df = pd.read_csv('sales_data.csv')
months = df['month_number'].values
ex7()
