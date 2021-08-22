def ex1():
    x = int(input('x: '))
    y = int(input('y: '))

    p = x * y

    if p <= 1000:
        print(p)
    else:
        print(x + y)


def ex2():
    r = 5
    prev = 0
    for i in range(r):
        summ = i + prev
        print(summ)
        prev = i


def ex3():
    list_a = [1, 2, 3, 4]
    if list_a[0] == list_a[-1]:
        print(True)


def ex4():
    list_a = list(range(20))
    for el in list_a:
        if (el % 5) == 0:
            print(el)


def ex5():
    phrase = 'la mama de la mama de la mama de la mama de la mama de la mama de la mama de la mama'
    tmp = 'mama'
    count = 0

    for i in range((len(phrase) - 1)):
        if phrase[i: i+4] == "mama":
            count += 1

    print(f'\'mama\' showed up {phrase.count(tmp)} times')


def ex6():
    list_a = [1, 3, 5, 6, 32, 775, 24, 12, 23, 67, 80, 18]
    list_b = [0, 21, 5, 6, 8, 7, 9, 102, 54, 55, 1, 89, 67]
    list_c = []

    for element in list_a:
        if (element % 2) != 0:
            list_c.append(element)

    for element in list_b:
        if (element % 2) == 0:
            list_c.append(element)

    print(list_c)


def ex7():
    str_1 = 'puta'
    str_2 = 'conchas'
    middle_i = int(len(str_1) / 2)
    print(type(middle_i))
    s3 = str_1[: middle_i] + str_2 + str_1[middle_i:]
    print(s3)


def ex8():
    str_1 = 'putas'
    str_2 = 'conchas'
    middle_i1 = int(len(str_1) / 2)
    middle_i2 = int(len(str_2) / 2)

    s3 = str_1[0] + str_2[0] + str_1[middle_i1] + str_2[middle_i2] + str_1[-1] + str_2[-1]

    print(s3, ' ', type(s3))


def ex9():
    stringa = 'tu Mama Es 1 gran 2819 @][]_! ewj h7v8 @à'
    cntUp = 0
    cntLow = 0
    cntD = 0
    cntS = 0

    for character in stringa:
        if character.isupper():
            cntUp += 1
        elif character.islower():
            cntLow += 1
        elif character.isnumeric():
            cntD += 1
        else:
            cntS += 1

    print(f'Upper: {cntUp}\nLower: {cntLow}\nDigit: {cntD}\nSpecial: {cntS}')


def ex10():
    input_string = 'Welcome to USA. Awesome usa, isn’t it? usausaUSa'
    tofind = 'usa'

    print((input_string.casefold()).count(tofind.casefold()))


def ex11():
    webada = '10 wenadas si p3s conch4tuherm4n4'
    w_list = webada.split()
    print(w_list)
    n_list = [int(character) for word in w_list for character in word if character.isnumeric()]
    print(n_list)

    sumtot = sum(n_list)
    avgg = sumtot / len(n_list)

    print('avg: {0}'.format(avgg))


def ex12():
    occ_dict = {}
    pendejada = 'laweafomepoweonfirmetasbienlocardomano'
    for char in pendejada:
        count = pendejada.count(char)
        occ_dict[char] = count

    print(occ_dict)


ex12()




