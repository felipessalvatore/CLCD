import numpy as np
from util import get_n_different_items, get_new_item
from util import Rt, Rt_eq, list2coordination, create_csv
from util import Rt_pt, Rt_eq_pt

from vocab import male_names, female_names, cities_and_states, countries
from vocab_pt import male_names_pt, female_names_pt, cities_pt, countries_pt


def contradiction_instance_1(person_list,
                             place_list,
                             n,
                             Rt_function=Rt,
                             Rt_eq_function=Rt_eq,
                             and_str="and"):
    """
    T = {x1 > x2, x2 > x3, ... , xn-1 > xn}

    new = xj > xi (i<j) ----------- 1
    """
    people = get_n_different_items(person_list, n + 1)
    chain = []
    for i in range(n):
        chain.append(Rt_function(people[i], people[i + 1]))
    sentence1 = " , ".join(chain)
    id_base = np.random.choice(range(n))
    id_bigger = np.random.choice(range(id_base + 1, n + 1))
    sentence2 = Rt_function(people[id_bigger], people[id_base])
    return sentence1, sentence2, 1


def contradiction_instance_2(person_list,
                             place_list,
                             n,
                             Rt_function=Rt,
                             Rt_eq_function=Rt_eq,
                             and_str="and"):
    """
    T = {x1 >= x2, x2 >= x3, ... , xn-1 >= xn
        xn > y}

    new = y > xi  ----------- 1
    """
    people = get_n_different_items(person_list, n + 1)
    new_person = get_new_item(people, person_list)
    chain = []
    for i in range(n):
        chain.append(Rt_eq_function(people[i], people[i + 1]))
    sentence1 = " , ".join(chain)
    sentence1 += " , " + Rt_function(people[-1], new_person)
    id_base = np.random.choice(range(n + 1))
    sentence2 = Rt_function(new_person, people[id_base])
    return sentence1, sentence2, 1


def contradiction_instance_3(person_list,
                             place_list,
                             n,
                             Rt_function=Rt,
                             Rt_eq_function=Rt_eq,
                             and_str="and"):
    """
    T = {x > [x1, ....., xn], x >= y}

    new = xi > y ----------- 1
    """
    people = get_n_different_items(person_list, n + 1)
    new_person = get_new_item(people, person_list)
    chain = []
    for i in range(n):
        chain.append(Rt_function(people[i], people[i + 1]))
    sentence1 = Rt_function(people[0], list2coordination(people[1:], and_str))
    eq = [Rt_eq_function(people[0], new_person), Rt_eq_function(new_person, people[0])]  # noqa
    eq = np.random.choice(eq)
    sentence1 += " , " + eq
    selected = np.random.choice(people[1:])
    sentence2 = Rt_function(selected, new_person)
    return sentence1, sentence2, 1


def non_contradiction_instance_1(person_list,
                                 place_list,
                                 n,
                                 Rt_function=Rt,
                                 Rt_eq_function=Rt_eq,
                                 and_str="and"):
    """
    T = {x1 > x2, x2 > x3, ... , xn-1 > xn}

    new = xi > xj (i<j) ----------- 0
    """
    people = get_n_different_items(person_list, n + 1)
    chain = []
    for i in range(n):
        chain.append(Rt_function(people[i], people[i + 1]))
    sentence1 = " , ".join(chain)
    id_base = np.random.choice(range(n))
    id_bigger = np.random.choice(range(id_base + 1, n + 1))
    sentence2 = Rt_function(people[id_base], people[id_bigger])
    return sentence1, sentence2, 0


def non_contradiction_instance_2(person_list,
                                 place_list,
                                 n,
                                 Rt_function=Rt,
                                 Rt_eq_function=Rt_eq,
                                 and_str="and"):
    """
    T = {x1 >= x2, x2 >= x3, ... , xn-1 >= xn,
         xn > y}

    new = xi > y  ----------- 0
    """
    people = get_n_different_items(person_list, n + 1)
    new_person = get_new_item(people, person_list)
    chain = []
    for i in range(n):
        chain.append(Rt_eq_function(people[i], people[i + 1]))
    sentence1 = " , ".join(chain)
    sentence1 += " , " + Rt_function(people[-1], new_person)
    id_base = np.random.choice(range(n + 1))
    sentence2 = Rt_function(people[id_base], new_person)
    return sentence1, sentence2, 0


def non_contradiction_instance_3(person_list,
                                 place_list,
                                 n,
                                 Rt_function=Rt,
                                 Rt_eq_function=Rt_eq,
                                 and_str="and"):
    """
    T = {x > [x1, ....., xn], x >= y}

    new = y > xi ----------- 0
    """
    people = get_n_different_items(person_list, n + 1)
    new_person = get_new_item(people, person_list)
    chain = []
    for i in range(n):
        chain.append(Rt_function(people[i], people[i + 1]))
    sentence1 = Rt_function(people[0], list2coordination(people[1:], and_str))
    eq = [Rt_eq_function(people[0], new_person), Rt_eq_function(new_person, people[0])]  # noqa
    eq = np.random.choice(eq)
    sentence1 += " , " + eq
    selected = np.random.choice(people[1:])
    sentence2 = Rt_function(new_person, selected)
    return sentence1, sentence2, 0


def eng2pt(f):
    return lambda x, y, z: f(x, y, z, Rt_function=Rt_pt, Rt_eq_function=Rt_eq_pt, and_str="e")  # noqa


positive_instances_list_en = [contradiction_instance_1,
                              contradiction_instance_2,
                              contradiction_instance_3]

negative_instances_list_en = [non_contradiction_instance_1,
                              non_contradiction_instance_2,
                              non_contradiction_instance_3]

positive_instances_list_pt = [eng2pt(f) for f in positive_instances_list_en]
negative_instances_list_pt = [eng2pt(f) for f in negative_instances_list_en]


if __name__ == '__main__':
    # call this script in the main folder, i.e., type
    # python clcd/text_generation/comparative.py

    # english
    create_csv(out_path="text_gen_output/comparative_train.csv",  # noqa
               size=10000,
               person_list=male_names,
               place_list=countries,
               min_n=4,
               n=10,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    create_csv(out_path="text_gen_output/comparative_test.csv",  # noqa
               size=1000,
               person_list=female_names,
               place_list=cities_and_states,
               min_n=4,
               n=10,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    # portuguese
    create_csv(out_path="text_gen_output/comparative_pt_train.csv",  # noqa
               size=10000,
               person_list=male_names_pt,
               place_list=countries_pt,
               min_n=4,
               n=10,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa

    create_csv(out_path="text_gen_output/comparative_pt_test.csv",  # noqa
               size=1000,
               person_list=female_names_pt,
               place_list=cities_pt,
               min_n=4,
               n=10,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa
