import array


def ex1():
    list_one = [3, 6, 9, 12, 15, 18, 21]
    list_two = [4, 8, 12, 16, 20, 24, 28]

    odd_el = list_one[1::2]
    even_el = list_two[0::2]

    list_three = odd_el + even_el
    print(list_three)


def ex2():
    in_list = [2, 34, 112, 543, 12, 3, 12, 13, 5, 6]
    element = in_list.pop(4)
    in_list.insert(1, element)
    in_list.append(element)

    print(in_list)


def ex3():
    in_list = [2, 34, 112, 543, 12, 3, 12, 13, 5]
    n = int(len(in_list) / 3)
    a = in_list[:n]
    b = in_list[n:2*n]
    c = in_list[2*n:]

    print(a, b, c)

    a.reverse()
    b.reverse()
    c.reverse()

    print(a, b, c)


def ex4():
    sample_list = [11, 45, 8, 11, 23, 45, 23, 45, 89]
    occ_dict = dict()
    for number in sample_list:
        if number in occ_dict:
            occ_dict[number] += 1
        else:
            occ_dict[number] = 1

    print(occ_dict)


def ex5():

    list1 = [2, 3, 4, 5, 6, 7, 8]
    list2 = [4, 9, 16, 25, 36, 49, 64]

    result = zip(list1, list2)
    result_set = set(result)
    print(result_set)


def ex6():
    firstSet = {23, 42, 65, 57, 78, 83, 29}
    secondSet = {57, 83, 29, 67, 73, 43, 48}

    intersect = firstSet.intersection(secondSet)
    for item in intersect:
        firstSet.remove(item)

    # firstSet.difference_update(secondSet)
    print(firstSet)


def ex7():
    set1 = {57, 83, 29}
    set2 = {57, 83, 29, 67, 73, 43, 48}

    if set2.issuperset(set1):
        print('Set 2 is Superset of Set 1')
    if set1.issubset(set2):
        set1.clear()
    if set2.issubset(set1):
        set2.clear()

    print('First set:', set1)
    print('Second set:', set2)


def ex8():
    rollNumber = [47, 64, 69, 37, 76, 83, 95, 97]
    sampleDict = {'Jhon': 47, 'Emma': 69, 'Kelly': 76, 'Jason': 97}

    i = 0
    for number in rollNumber:
        if number in sampleDict.values():
            rollNumber[i] = number
            i += 1
    del rollNumber[i:]
    print(rollNumber)
    # rollNumber[:] = [item for item in rollNumber if item in sampleDict.values()]


def ex9():
    speed = {'Jan': 47, 'Feb': 52, 'March': 47, 'April': 44, 'May': 52, 'June': 53, 'July': 54, 'Aug': 44, 'Sept': 54}
    listi = list()
    for elem in speed.values():
        if elem not in listi:
            listi.append(elem)

    print(listi)


def ex10():
    sample_list = [87, 52, 44, 53, 54, 87, 52, 53]
    print(sample_list)
    sample_list = list(set(sample_list))
    print(sample_list)
    tup = tuple(sample_list)
    maxx = max(tup)
    minn = min(tup)
    print('Max is: ', maxx, '\nMin is: ', minn)


ex10()
