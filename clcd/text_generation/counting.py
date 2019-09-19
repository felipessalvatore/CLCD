import numpy as np

from vocab import male_names, female_names, cities_and_states, countries
from vocab_pt import male_names_pt, female_names_pt, cities_pt, countries_pt

from util import get_n_different_items
from util import vi, vi_On, person, not_vi, create_csv, num2word
from util import vi_pt, vi_On_pt, person_pt, not_vi_pt, num2word_pt


def contradiction_instance_1(person_list,
                             place_list,
                             n,
                             vi_function=vi,
                             vi_On_function=vi_On,
                             num2word_dict=num2word,
                             choices_plr=["only @ places",
                                          "only @ people"],
                             choices_sing=["only @ place",
                                           "only @ people"]):
    """
    T = x has visited only num2word(n) places/people
      new = x visited P1, ...., Pn+1/x1, ...., xn+1
    ----------- 1
    """

    people = np.random.choice(person_list)
    places = get_n_different_items(place_list, n + 1)
    new_people = get_n_different_items(person_list, n + 1)
    if n == 1:
        coin_choice = choices_sing
    else:
        coin_choice = choices_plr
    n = num2word_dict[n]
    coin = np.random.choice([0, 1])
    if coin == 0:
        cdr = coin_choice[0]
        cdr = cdr.replace("@", n)
        sentence1 = vi_function(people, cdr)
        sentence2 = vi_On_function(people, places)
    else:
        cdr = coin_choice[1]
        if cdr.find("pessoas") != -1 and n == "dois":  # hack
            n = "duas"  # hack
        cdr = cdr.replace("@", n)
        sentence1 = vi_function(people, cdr)
        sentence2 = vi_On_function(people, new_people)
    return sentence1, sentence2, 1


def contradiction_instance_2(person_list,
                             place_list,
                             n,
                             vi_function=vi,
                             vi_On_function=vi_On,
                             num2word_dict=num2word,
                             place_vrs=["only @ place", "only @ places"],
                             people_vrs=["only @ people", "only @ people"],
                             coord=" and "):
    """
    T = x has visited only num2word(n) place and only num2word(m) people
      new = x visited x1, ...., xm+1
      or
      new = x visited P1, ...., Pn+1
    ----------- 1
    """
    m = np.random.choice(range(1, n + 1))
    n_ = num2word_dict[n]
    m_ = num2word_dict[m]
    people = np.random.choice(person_list)
    if n == 1:
        place_str = place_vrs[0]
    else:
        place_str = place_vrs[1]
    if m == 1:
        people_str = people_vrs[0]
    else:
        people_str = people_vrs[1]

    place_str = place_str.replace("@", n_)

    if people_str.find("pessoas") != -1 and m_ == "dois":  # hack
        m_ = "duas"  # hack
    if people_str.find("pessoa") != -1 and m_ == "um":  # hack
        m_ = "uma"  # hack

    people_str = people_str.replace("@", m_)

    and_str = place_str + coord + people_str
    sentence1 = vi_function(people, and_str)
    coin = np.random.choice([0, 1])
    if coin == 0:
        new_people = get_n_different_items(person_list, m + 1)
        sentence2 = vi_On_function(people, new_people)
    else:
        places = get_n_different_items(place_list, n + 1)
        sentence2 = vi_On_function(people, places)
    return sentence1, sentence2, 1


def non_contradiction_instance_1(person_list,
                                 place_list,
                                 n,
                                 vi_function=vi,
                                 vi_On_function=vi_On,
                                 num2word_dict=num2word,
                                 choices_plr=["only @ places",
                                              "only @ people"],
                                 choices_sing=["only @ place",
                                               "only @ people"]):
    """
    T = x has visited only num2word(n) countries/people
      new = x visited P1, ...., Pk/x1, ...., xk (k<n)
    ----------- 0
    """
    if n == 1:
        k = 1
    else:
        k = np.random.choice(range(1, n))
    people = np.random.choice(person_list)
    places = get_n_different_items(place_list, k)
    new_people = get_n_different_items(person_list, k)
    if n == 1:
        coin_choice = choices_sing
    else:
        coin_choice = choices_plr
    n = num2word_dict[n]
    coin = np.random.choice([0, 1])
    if coin == 0:
        cdr = coin_choice[0]
        cdr = cdr.replace("@", n)
        sentence1 = vi_function(people, cdr)
        if k == 1:
            sentence2 = vi_function(people, places[0])
        else:
            sentence2 = vi_On_function(people, places)
    else:
        cdr = coin_choice[1]
        if cdr.find("pessoas") != -1 and n == "dois":  # hack
            n = "duas"  # hack
        cdr = cdr.replace("@", n)
        sentence1 = vi_function(people, cdr)
        if k == 1:
            sentence2 = vi_function(people, new_people[0])
        else:
            sentence2 = vi_On_function(people, new_people)
    return sentence1, sentence2, 0


def non_contradiction_instance_2(person_list,
                                 place_list,
                                 n,
                                 vi_function=vi,
                                 vi_On_function=vi_On,
                                 num2word_dict=num2word,
                                 place_vrs=["only @ place", "only @ places"],
                                 people_vrs=["only @ people", "only @ people"],
                                 coord=" and "):
    """
    T = x has visited only num2word(n) countries and only num2word(m) people
      new = x visited x1, ...., xk (k<n)
      or
      new = x visited P1, ...., Pk (k<m)
    ----------- 0
    """
    m = np.random.choice(range(1, n + 1))
    n_ = num2word_dict[n]
    m_ = num2word_dict[m]
    people = np.random.choice(person_list)
    if n == 1:
        place_str = place_vrs[0]
    else:
        place_str = place_vrs[1]
    if m == 1:
        people_str = people_vrs[0]
    else:
        people_str = people_vrs[1]

    place_str = place_str.replace("@", n_)

    if people_str.find("pessoas") != -1 and m_ == "dois":  # hack
        m_ = "duas"  # hack
    if people_str.find("pessoa") != -1 and m_ == "um":  # hack
        m_ = "uma"  # hack

    people_str = people_str.replace("@", m_)

    and_str = place_str + coord + people_str
    sentence1 = vi_function(people, and_str)
    coin = np.random.choice([0, 1])
    if coin == 0:
        if m == 1:
            k = 1
        else:
            k = np.random.choice(range(1, m))
        new_people = get_n_different_items(person_list, k)
        sentence2 = vi_On_function(people, new_people)
    else:
        if n == 1:
            k = 1
        else:
            k = np.random.choice(range(1, n))
        places = get_n_different_items(place_list, k)
        sentence2 = vi_On_function(people, places)
    return sentence1, sentence2, 0


def eng2pt_5(f, g1, g2, g3, g4, g5):
    return lambda x, y, z: f(x, y, z, g1, g2, g3, g4, g5)  # noqa


def eng2pt_6(f, g1, g2, g3, g4, g5, g6):
    return lambda x, y, z: f(x, y, z, g1, g2, g3, g4, g5, g6)  # noqa


contradiction_instance_1_pt = eng2pt_5(contradiction_instance_1,
                                       vi_pt,
                                       vi_On_pt,
                                       num2word_pt,
                                       ["apenas @ lugares",
                                        "apenas @ pessoas"],
                                       ["apenas @ lugar",
                                        "apenas @a pessoa"])


contradiction_instance_2_pt = eng2pt_6(contradiction_instance_2,
                                       vi_pt,
                                       vi_On_pt,
                                       num2word_pt,
                                       ["apenas @ lugar", "apenas @ lugares"],
                                       ["apenas @ pessoa", "apenas @ pessoas"],
                                       " e ")


non_contradiction_instance_1_pt = eng2pt_5(non_contradiction_instance_1,
                                           vi_pt,
                                           vi_On_pt,
                                           num2word_pt,
                                           ["apenas @ lugares",
                                            "apenas @ pessoas"],
                                           ["apenas @ lugar",
                                               "apenas @a pessoa"])

non_contradiction_instance_2_pt = eng2pt_6(non_contradiction_instance_2,
                                           vi_pt,
                                           vi_On_pt,
                                           num2word_pt,
                                           ["apenas @ lugar", "apenas @ lugares"],  # noqa
                                           ["apenas @ pessoa", "apenas @ pessoas"],  # noqa
                                           " e ")


positive_instances_list_en = [contradiction_instance_1,
                              contradiction_instance_2]

negative_instances_list_en = [non_contradiction_instance_1,
                              non_contradiction_instance_2]


positive_instances_list_pt = [contradiction_instance_1_pt,
                              contradiction_instance_2_pt]

negative_instances_list_pt = [non_contradiction_instance_1_pt,
                              non_contradiction_instance_2_pt]

if __name__ == '__main__':
    # call this script in the main folder, i.e., type
    # python clcd/text_generation/counting.py

    # english
    create_csv(out_path="fixed_data/contra_scale/counting_train.csv",  # noqa
               size=10000,
               person_list=male_names,
               place_list=countries,
               min_n=1,
               n=30,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    create_csv(out_path="fixed_data/contra_scale/counting_test.csv",  # noqa
               size=1000,
               person_list=female_names,
               place_list=cities_and_states,
               min_n=1,
               n=30,
               positive_instances_list=positive_instances_list_en,
               negative_instances_list=negative_instances_list_en)  # noqa

    # portuguese
    create_csv(out_path="fixed_data/contra_scale/counting_pt_train.csv",  # noqa
               size=10000,
               person_list=male_names_pt,
               place_list=countries_pt,
               min_n=1,
               n=30,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa

    create_csv(out_path="fixed_data/contra_scale/counting_pt_test.csv",  # noqa
               size=1000,
               person_list=female_names_pt,
               place_list=cities_pt,
               min_n=1,
               n=30,
               positive_instances_list=positive_instances_list_pt,
               negative_instances_list=negative_instances_list_pt)  # noqa
