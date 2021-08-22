import json


def ex1():
    json_obj = '{"Name": "David", "Class": "I", "Age": 6}'
    python_obj = json.loads(json_obj)

    print(python_obj)
    print(python_obj['Name'])
    print(python_obj['Class'])
    print(python_obj['Age'])


def ex2():
    python_obj = {'Name': 'David',
                  'Class': 'I',
                  'Age': 6
                  }
    json_obj = json.dumps(python_obj)
    print(json_obj, 'type:', type(json_obj))


def ex4():
    python_obj = {'30': 5, '1': 2, '31': 90}
    json_obj = json.dumps(python_obj, sort_keys=True, indent=4)
    print(json_obj)


def ex5():
    with open('states.json') as f:
        state_data = json.load(f)
    for state in state_data['states']:
        del state['area_codes']
    with open('new_states.json', 'w') as f:
        for state in state_data:
            json.dump(state_data, f, indent=2)


ex5()
