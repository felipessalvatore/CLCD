import numpy as np
from util import get_new_item, get_n_different_items
from util import vi_Sn, vi_On, not_vi, create_csv

from vocab import male_names, female_names, cities_and_states, countries
from vocab_pt import male_names_pt, female_names_pt, cities_pt, countries_pt

from util import vi_Sn_pt, vi_On_pt, not_vi_pt


def contradiction_instance_1(person_list,
                             place_list,
                             n,
                             vi_Sn_function=vi_Sn,
                             not_vi_function=not_vi):
    """
    T = {v([x1,...,xn], P)}
    new = not v(xi,P) ----------- 1
    """
    people = get_n_different_items(person_list, n)
    place = np.random.choice(place_list)
    sentence1 = vi_Sn_function(people, place)
    selected = np.random.choice(people)
    sentence2 = not_vi_function(selected, place)
    return sentence1, sentence2, 1


def contradiction_instance_2(person_list,
                             place_list,
                             n,
                             vi_On_function=vi_On,
                             not_vi_function=not_vi):
    """
    T = {v(x, [P1,...,Pn])}
    new = not v(x,Pi) ----------- 1
    """
    people = np.random.choice(person_list)
    places = get_n_different_items(place_list, n)
    sentence1 = vi_On_function(people, places)
    selected = np.random.choice(places)
    sentence2 = not_vi_function(people, selected)
    return sentence1, sentence2, 1


def non_contradiction_instance_1(person_list,
                                 place_list,
                                 n,
                                 vi_Sn_function=vi_Sn,
                                 not_vi_function=not_vi):
    """
    T = {v([x1,...,xn], P)}
    new = not v(xi,P*)
    or
    new = not v(x*,P)
    P* not in {P}
    x* not in {x1, ..., xn} ----------- 0
    """
    people = get_n_different_items(person_list, n)
    place = np.random.choice(place_list)
    sentence1 = vi_Sn_function(people, place)
    new_person = get_new_item(people, person_list)
    new_place = get_new_item([place], place_list)
    selected = np.random.choice(people)
    coin = np.random.choice([0, 1])
    if coin == 0:
        sentence2 = not_vi_function(selected, new_place)
    else:
        sentence2 = not_vi_function(new_person, place)
    return sentence1, sentence2, 0


def non_contradiction_instance_2(person_list,
                                 place_list,
                                 n,
                                 vi_On_function=vi_On,
                                 not_vi_function=not_vi):
    """
    T = {v(x, [P1,...,Pn])}
    new = not v(x*,Pi)
    or
    new = not v(x,P*)
    P* not in {P1,...,Pn}
    x* not in {x} ----------- 0
    """
    people = np.random.choice(person_list)
    places = get_n_different_items(place_list, n)
    sentence1 = vi_On_function(people, places)
    new_person = get_new_item([people], person_list)
    new_place = get_new_item(places, place_list)
    selected = np.random.choice(places)
    coin = np.random.choice([0, 1])
    if coin == 0:
        sentence2 = not_vi_function(new_person, selected)
    else:
        sentence2 = not_vi_function(people, new_place)
    return sentence1, sentence2, 0


def eng2pt(f, g, h):
    return lambda x, y, z: f(x, y, z, g, h)  # noqa


positive_instances_list_en = [contradiction_instance_1,
                              contradiction_instance_2]

negative_instances_list_en = [non_contradiction_instance_1,
                              non_contradiction_instance_2]


positive_instances_list_pt = [eng2pt(contradiction_instance_1, vi_Sn_pt, not_vi_pt), # noqa
                              eng2pt(contradiction_instance_2, vi_On_pt, not_vi_pt)] # noqa

negative_instances_list_pt = [eng2pt(non_contradiction_instance_1, vi_Sn_pt, not_vi_pt), # noqa
                              eng2pt(non_contradiction_instance_2, vi_On_pt, not_vi_pt)] # noqa


if __name__ == '__main__':

    # call this script in the main folder, i.e., type
    # python clcd/text_generation/boolean_coordination.py

    # english
    create_csv(out_path="text_gen_output/boolean_coordination_train.csv",  # noqa
               size=10000,
               person_list=male_names,
               place_list=countries,
               min_n=2,
               n=20,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    create_csv(out_path="text_gen_output/boolean_coordination_test.csv",  # noqa
               size=1000,
               person_list=female_names,
               place_list=cities_and_states,
               min_n=2,
               n=20,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    # portuguese
    create_csv(out_path="text_gen_output/boolean_coordination_pt_train.csv",  # noqa
               size=10000,
               person_list=male_names_pt,
               place_list=countries_pt,
               min_n=2,
               n=20,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa

    create_csv(out_path="text_gen_output/boolean_coordination_pt_test.csv",  # noqa
               size=1000,
               person_list=female_names_pt,
               place_list=cities_pt,
               min_n=2,
               n=20,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa