import numpy as np

from vocab import male_names, female_names, cities_and_states, countries
from vocab_pt import male_names_pt, female_names_pt, cities_pt, countries_pt

from util import get_n_different_items
from util import vi, person, not_vi, create_csv
from util import vi_pt, person_pt, not_vi_pt


def contradiction_instance_1(person_list,
                             place_list,
                             n,
                             person_function=person,
                             vi_function=vi,
                             not_vi_function=not_vi,
                             every_str="every place"):
    """
    T{ x is the person that has visisted every place
      v(x, y)}
    new = not v(x,Pi) ----------- 1
    """

    people = get_n_different_items(person_list, 2)
    place = np.random.choice(place_list)
    eq = person_function(people[0], lambda x: vi_function(x, every_str))
    visit = [vi_function(people[0], people[1]), vi_function(people[1], people[0])] # noqa
    visit = np.random.choice(visit)
    coin = np.random.choice([0, 1])
    if coin == 0:
        sentence1 = visit + ", " + eq
    else:
        sentence1 = eq + ", " + visit
    sentence2 = not_vi_function(people[0], place)
    return sentence1, sentence2, 1


def contradiction_instance_2(person_list,
                             place_list,
                             n,
                             person_function=person,
                             vi_function=vi,
                             not_vi_function=not_vi,
                             everyone_str="everyone"):
    """
    T{ x is the person that has visisted everyone
      v(x, y)}
    new = not v(x, z) ----------- 1
    """

    people = get_n_different_items(person_list, 3)
    eq = person_function(people[0], lambda x: vi_function(x, everyone_str))
    visit = [vi_function(people[0], people[1]), vi_function(people[1], people[0])] # noqa
    visit = np.random.choice(visit)
    coin = np.random.choice([0, 1])
    if coin == 0:
        sentence1 = visit + ", " + eq
    else:
        sentence1 = eq + ", " + visit
    sentence2 = not_vi_function(people[0], people[2])
    return sentence1, sentence2, 1


def non_contradiction_instance_1(person_list,
                                 place_list,
                                 n,
                                 person_function=person,
                                 vi_function=vi,
                                 not_vi_function=not_vi,
                                 every_str="every place"):
    """
    T{ x is the person that has visisted every place
      v(x, y)}
    new = not v(y,Pi) ----------- 0
    """

    people = get_n_different_items(person_list, 2)
    place = np.random.choice(place_list)
    eq = person_function(people[0], lambda x: vi_function(x, every_str))
    visit = [vi_function(people[0], people[1]), vi_function(people[1], people[0])] # noqa
    visit = np.random.choice(visit)
    coin = np.random.choice([0, 1])
    if coin == 0:
        sentence1 = visit + ", " + eq
    else:
        sentence1 = eq + ", " + visit
    sentence2 = not_vi_function(people[1], place)
    return sentence1, sentence2, 0


def non_contradiction_instance_2(person_list,
                                 place_list,
                                 n,
                                 person_function=person,
                                 vi_function=vi,
                                 not_vi_function=not_vi,
                                 everyone_str="everyone"):
    """
    T{ x is the person that has visisted everyone
      v(x, y)}
    new = not v(y, z) ----------- 0
    """

    people = get_n_different_items(person_list, 3)
    eq = person_function(people[0], lambda x: vi_function(x, everyone_str))
    visit = [vi_function(people[0], people[1]), vi_function(people[1], people[0])] # noqa
    visit = np.random.choice(visit)
    coin = np.random.choice([0, 1])
    if coin == 0:
        sentence1 = visit + ", " + eq
    else:
        sentence1 = eq + ", " + visit
    sentence2 = not_vi_function(people[1], people[2])
    return sentence1, sentence2, 0


def eng2pt(f, g, h, i, j):
    return lambda x, y, z: f(x, y, z, g, h, i, j)  # noqa


positive_instances_list_en = [contradiction_instance_1,
                              contradiction_instance_2]

negative_instances_list_en = [non_contradiction_instance_1,
                              non_contradiction_instance_2]


positive_instances_list_pt = [eng2pt(contradiction_instance_1,
                                     person_pt,
                                     vi_pt,
                                     not_vi_pt,
                                     "todo lugar"),
                              eng2pt(contradiction_instance_2,
                                     person_pt,
                                     vi_pt,
                                     not_vi_pt,
                                     "todo mundo")]

negative_instances_list_pt = [eng2pt(non_contradiction_instance_1,
                                     person_pt,
                                     vi_pt,
                                     not_vi_pt,
                                     "todo lugar"),
                              eng2pt(non_contradiction_instance_2,
                                     person_pt,
                                     vi_pt,
                                     not_vi_pt,
                                     "todo mundo")]

if __name__ == '__main__':
    # call this script in the main folder, i.e., type
    # python clcd/text_generation/equality_and_definite_description.py

    # english
    create_csv(out_path="fixed_data/contra_scale/equality_and_definite_description_train.csv",  # noqa
               size=10000,
               person_list=male_names,
               place_list=countries,
               min_n=2,
               n=2,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    create_csv(out_path="fixed_data/contra_scale/equality_and_definite_description_test.csv",  # noqa
               size=1000,
               person_list=female_names,
               place_list=cities_and_states,
               min_n=2,
               n=2,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    # portuguese
    create_csv(out_path="fixed_data/contra_scale/equality_and_definite_description_pt_train.csv",  # noqa
               size=10000,
               person_list=male_names_pt,
               place_list=countries_pt,
               min_n=2,
               n=2,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa

    create_csv(out_path="fixed_data/contra_scale/equality_and_definite_description_pt_test.csv",  # noqa
               size=1000,
               person_list=female_names_pt,
               place_list=cities_pt,
               min_n=2,
               n=2,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa