import numpy as np
from vocab import male_names, female_names, cities_and_states, countries
from vocab_pt import male_names_pt, female_names_pt, cities_pt, countries_pt

from util import get_new_item, get_n_different_items
from util import vi, not_vi, vi_pt, not_vi_pt, create_csv


def contradiction_instance_1(person_list,
                             place_list,
                             n,
                             vi_function=vi,
                             not_vi_function=not_vi):
    """
    T = {v(x1,P1), ..., v(xn,Pn)}
    new = not v(xi,Pi) ----------- 1
    """
    people = get_n_different_items(person_list, n)
    places = get_n_different_items(place_list, n)
    sentence1 = ", ".join([vi_function(x, y) for x, y in zip(people, places)])
    id_ = np.random.choice(len(people))
    sentence2 = not_vi_function(people[id_], places[id_])
    return sentence1, sentence2, 1


def non_contradiction_instance_1(person_list,
                                 place_list,
                                 n,
                                 vi_function=vi,
                                 not_vi_function=not_vi):
    """
    T = {v(x1,P1), ..., v(xn,Pn)}
    new = not v(xi,P*) ----------- 0
    P* not in {P1, ..., Pn}
    """
    people = get_n_different_items(person_list, n)
    places = get_n_different_items(place_list, n)
    sentence1 = ", ".join([vi_function(x, y) for x, y in zip(people, places)])
    id_ = np.random.choice(len(people))
    new_place = get_new_item(places, place_list)
    sentence2 = not_vi_function(people[id_], new_place)
    return sentence1, sentence2, 0


def non_contradiction_instance_2(person_list,
                                 place_list,
                                 n,
                                 vi_function=vi,
                                 not_vi_function=not_vi):
    """
    T = {v(x1,P1), ..., v(xn,Pn)}
    new = not v(x*,Pi) ----------- 0
    x* not in {x1, ..., xn}
    """
    people = get_n_different_items(person_list, n)
    places = get_n_different_items(place_list, n)
    sentence1 = ", ".join([vi_function(x, y) for x, y in zip(people, places)])
    id_ = np.random.choice(len(people))
    new_person = get_new_item(people, person_list)
    sentence2 = not_vi_function(new_person, places[id_])
    return sentence1, sentence2, 0


def eng2pt(f):
    return lambda x, y, z: f(x, y, z, vi_function=vi_pt, not_vi_function=not_vi_pt)  # noqa


positive_instances_list_en = [contradiction_instance_1]
negative_instances_list_en = [non_contradiction_instance_1,
                              non_contradiction_instance_2]


positive_instances_list_pt = [eng2pt(f) for f in positive_instances_list_en]
negative_instances_list_pt = [eng2pt(f) for f in negative_instances_list_en]


if __name__ == '__main__':

    # call this script in the main folder, i.e., type
    # python clcd/text_generation/simple_negation.py

    # english
    create_csv(out_path="fixed_data/contra_scale/simple_negation_train.csv",  # noqa
               size=10000,
               person_list=male_names,
               place_list=countries,
               min_n=2,
               n=12,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa
    create_csv(out_path="fixed_data/contra_scale/simple_negation_test.csv",  # noqa
               size=1000,
               person_list=female_names,
               place_list=cities_and_states,
               min_n=2,
               n=12,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    # portuguese
    create_csv(out_path="fixed_data/contra_scale/simple_negation_pt_train.csv",  # noqa
               size=10000,
               person_list=male_names_pt,
               place_list=countries_pt,
               min_n=2,
               n=12,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa
    create_csv(out_path="fixed_data/contra_scale/simple_negation_pt_test.csv",  # noqa
               size=1000,
               person_list=female_names_pt,
               place_list=cities_pt,
               min_n=2,
               n=12,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa
