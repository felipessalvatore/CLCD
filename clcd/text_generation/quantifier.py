import numpy as np
from util import get_new_item, get_n_different_items
from util import vi, vi_On, not_vi, create_csv
from util import vi_pt, vi_On_pt, not_vi_pt

from vocab import male_names, female_names, cities_and_states, countries
from vocab_pt import male_names_pt, female_names_pt, cities_pt, countries_pt


def contradiction_instance_1(person_list,
                             place_list,
                             n,
                             vi_On_function=vi_On,
                             not_vi_function=not_vi,
                             Everyone_str="Everyone"):
    """
    T = {every x v(x,[P1, ..., Pn]}
    new = not v(xi,Pi) ----------- 1
    """
    people = np.random.choice(person_list)
    places = get_n_different_items(place_list, n)
    sentence1 = vi_On_function(Everyone_str, places)
    selected = np.random.choice(places)
    sentence2 = not_vi_function(people, selected)
    return sentence1, sentence2, 1


def contradiction_instance_2(person_list,
                             place_list,
                             n,
                             vi_function=vi,
                             not_vi_function=not_vi,
                             Everyone_str="Everyone",
                             every_place_str="every place"):
    """
    T = {every x every P v(x,P)}
    new = not v(xi,Pi) ----------- 1
    """
    people = np.random.choice(person_list)
    place = np.random.choice(place_list)
    sentence1 = vi_function(Everyone_str, every_place_str)
    sentence2 = not_vi_function(people, place)
    return sentence1, sentence2, 1


def contradiction_instance_3(person_list,
                             place_list,
                             n,
                             vi_function=vi,
                             not_vi_function=not_vi,
                             Everyone_str="Everyone",
                             every_person_str="every person"):
    """
    T = {every x every y v(x,y)}
    new = not v(xi,xj) ----------- 1
    """
    people = get_n_different_items(person_list, 2)
    sentence1 = vi_function(Everyone_str, every_person_str)
    sentence2 = not_vi_function(people[0], people[1])
    return sentence1, sentence2, 1


def contradiction_instance_4(person_list,
                             place_list,
                             n,
                             vi_function=vi,
                             not_vi_function=not_vi,
                             Everyone_str="Everyone",
                             person_place="every person and every place"):
    """
    T = {every x every y every P v(x,y) and v(x,P)}
    new = not v(xi,xj)
    or
    new = not v(xi,Pj) ----------- 1
    """
    people = get_n_different_items(person_list, 2)
    place = np.random.choice(place_list)
    sentence1 = vi_function(Everyone_str, person_place)
    coin = np.random.choice([0, 1])
    if coin == 0:
        sentence2 = not_vi_function(people[0], people[1])
    else:
        sentence2 = not_vi_function(people[0], place)
    return sentence1, sentence2, 1


def non_contradiction_instance_1(person_list,
                                 place_list,
                                 n,
                                 vi_On_function=vi_On,
                                 not_vi_function=not_vi,
                                 Everyone_str="Everyone"):
    """
    T = { every x v(x1,[P1, ..., Pn]}
    new = not v(xi,P*) ----------- 0
    """
    people = np.random.choice(person_list)
    places = get_n_different_items(place_list, n)
    sentence1 = vi_On_function(Everyone_str, places)
    new_place = get_new_item(places, place_list)
    sentence2 = not_vi_function(people, new_place)
    return sentence1, sentence2, 0


def non_contradiction_instance_2(person_list,
                                 place_list,
                                 n,
                                 vi_function=vi,
                                 not_vi_function=not_vi,
                                 Everyone_str="Everyone",
                                 every_place_str="every place"):
    """
    T = {every x every P v(x,P)}
    new = not v(xi, xj) ----------- 0
    """
    people = get_n_different_items(person_list, 2)
    sentence1 = vi_function(Everyone_str, every_place_str)
    sentence2 = not_vi_function(people[0], people[1])
    return sentence1, sentence2, 0


def non_contradiction_instance_3(person_list,
                                 place_list,
                                 n,
                                 vi_function=vi,
                                 not_vi_function=not_vi,
                                 Everyone_str="Everyone",
                                 every_person_str="every person"):
    """
    T = {every x every y v(x,y)}
    new = not v(xi,Pj) ----------- 0
    """
    people = np.random.choice(person_list)
    place = np.random.choice(place_list)
    sentence1 = vi_function(Everyone_str, every_person_str)
    sentence2 = not_vi_function(people, place)
    return sentence1, sentence2, 0


def non_contradiction_instance_4(person_list,
                                 place_list,
                                 n,
                                 vi_function=vi,
                                 not_vi_function=not_vi,
                                 Someone_str="Someone",
                                 person_place="every person and every place"):
    """
    T = {some x every y every P v(x,y) and v(x,P)}
    new = not v(xi,xj)
    or
    new = not v(xi,Pj) ----------- 0
    """
    people = get_n_different_items(person_list, 2)
    place = np.random.choice(place_list)
    sentence1 = vi_function(Someone_str, person_place)
    coin = np.random.choice([0, 1])
    if coin == 0:
        sentence2 = not_vi_function(people[0], people[1])
    else:
        sentence2 = not_vi_function(people[0], place)
    return sentence1, sentence2, 0


def eng2pt_3(f, g1, g2, g3):
    return lambda x, y, z: f(x, y, z, g1, g2, g3)  # noqa


def eng2pt_4(f, g1, g2, g3, g4):
    return lambda x, y, z: f(x, y, z, g1, g2, g3, g4)  # noqa


contradiction_instance_1_pt = eng2pt_3(contradiction_instance_1,
                                       vi_On_pt,
                                       not_vi_pt,
                                       "Todo mundo")

contradiction_instance_2_pt = eng2pt_4(contradiction_instance_2,
                                       vi_pt,
                                       not_vi_pt,
                                       "Todo mundo",
                                       "todos os lugares")

contradiction_instance_3_pt = eng2pt_4(contradiction_instance_3,
                                       vi_pt,
                                       not_vi_pt,
                                       "Todo mundo",
                                       "todas as pessoas")


contradiction_instance_4_pt = eng2pt_4(contradiction_instance_4,
                                       vi_pt,
                                       not_vi_pt,
                                       "Todo mundo",
                                       "todas as pessoas e todos os lugares")

non_contradiction_instance_1_pt = eng2pt_3(non_contradiction_instance_1,
                                           vi_On_pt,
                                           not_vi_pt,
                                           "Todo mundo")


non_contradiction_instance_2_pt = eng2pt_4(non_contradiction_instance_2,
                                           vi_pt,
                                           not_vi_pt,
                                           "Todo mundo",
                                           "todos os lugares")

non_contradiction_instance_3_pt = eng2pt_4(non_contradiction_instance_3,
                                           vi_pt,
                                           not_vi_pt,
                                           "Todo mundo",
                                           "todas as pessoas")

non_contradiction_instance_4_pt = eng2pt_4(non_contradiction_instance_4,
                                           vi_pt,
                                           not_vi_pt,
                                           "Alguem",
                                           "todas as pessoas e todos os lugares")  # noqa

positive_instances_list_en = [contradiction_instance_1,
                              contradiction_instance_2,
                              contradiction_instance_3,
                              contradiction_instance_4]

negative_instances_list_en = [non_contradiction_instance_1,
                              non_contradiction_instance_2,
                              non_contradiction_instance_3,
                              non_contradiction_instance_4]

positive_instances_list_pt = [contradiction_instance_1_pt,
                              contradiction_instance_2_pt,
                              contradiction_instance_3_pt,
                              contradiction_instance_4_pt]

negative_instances_list_pt = [non_contradiction_instance_1_pt,
                              non_contradiction_instance_2_pt,
                              non_contradiction_instance_3_pt,
                              non_contradiction_instance_4_pt]

if __name__ == '__main__':
    # call this script in the main folder, i.e., type
    # python clcd/text_generation/quantifier.py

    # english
    create_csv(out_path="text_gen_output/quantifier_train.csv",  # noqa
               size=10000,
               person_list=male_names,
               place_list=countries,
               min_n=10,
               n=20,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    create_csv(out_path="text_gen_output/quantifier_test.csv",  # noqa
               size=1000,
               person_list=female_names,
               place_list=cities_and_states,
               min_n=10,
               n=20,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    # portuguese
    create_csv(out_path="text_gen_output/quantifier_pt_train.csv",  # noqa
               size=10000,
               person_list=male_names_pt,
               place_list=countries_pt,
               min_n=10,
               n=20,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa

    create_csv(out_path="text_gen_output/quantifier_pt_test.csv",  # noqa
               size=1000,
               person_list=female_names_pt,
               place_list=cities_pt,
               min_n=10,
               n=20,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa
