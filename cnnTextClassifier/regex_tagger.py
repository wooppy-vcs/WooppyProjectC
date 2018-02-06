import pprint
import re
from pymongo import MongoClient

import yaml

def open_yml_file(yml_file):
    """
    Open yaml file type as a config.
    :param yml_file:
    :return: a config object that can be access by tuple method. e.g. cfg['key'] = value
    """
    with open(yml_file, "r") as yml:
        cfg = yaml.load(yml)
    return cfg

"""
This method obtains the regex lists from MongoDB with variables as specified in mongo_connector.yml
Returns two dict objects - keyword_regex and scenario_keywords which together allows for parsing into
the right scenarios
"""
def read_regex_mongo():
    yml_file = "./mongo_connector.yml"
    cfg = open_yml_file(yml_file)
    client = MongoClient(cfg['mongodb']['endpoint'])
    db = client[cfg['mongodb']['database']]
    collection = db[cfg['mongodb']['collection']]
    document = collection.find_one()
    # pprint.pprint(document)
    keyword_regex = document[cfg['mongodb']['keyword_regex']]
    scenario_keywords = document[cfg['mongodb']['scenario_keywords']]
    return keyword_regex, scenario_keywords


"""
Takes in an input string and a keyword_dict_compiled_pattern to return a set of all the
keywords which matches
"""
def regex_tag(input_string, keyword_dict):
    keyword_set = set()
    for key,value in keyword_dict.items():
        for compiled_pattern in value:
            if(compiled_pattern.search(input_string)):
                keyword_set.add(key)
    return keyword_set

"""
This method compiles all the patterns found in the mongoDB document and presents it in a dict to be reused
"""
def create_keyword_dictionary(keyword_regex):
    keyword_dict_compiled_pattern = {}
    for key,value in keyword_regex.items():
        compiled_pattern_list = []
        for pattern in value:
            compiled_pattern = re.compile(pattern,re.IGNORECASE)
            compiled_pattern_list.append(compiled_pattern)
        keyword_dict_compiled_pattern[key] = compiled_pattern_list
    return keyword_dict_compiled_pattern

"""
This method compiles the keywords for each scenario into a set to enable comparison
"""
def create_scenario_dictionary(scenario_keywords):
    scenario_dictionary = {}
    for key,value in scenario_keywords.items():
        list_set = []
        for each_list in value:
            new_set = set(each_list)
            list_set.append(new_set)
        scenario_dictionary[key] = list_set
    return scenario_dictionary

"""
Compares found keywords with scenario dictionary. If the keyword set matches, will return only one scenario
Note: comments are if you want to return it as a list, which we decided not to do because it adds complication
Returns singleton None if nothing is tagged!
"""
def tag_scenario(tagged_keywords, scenario_dictionary):
    # list_scenarios = set()
    for key,value in scenario_dictionary.items():
        for each_list in value:
            if(each_list.issubset(tagged_keywords)):
                # list_scenarios.add(key)
                return key
    return "None"
    # return list_scenarios

"""
Should be run during the main method in order to create persistent dictionaries to be used without having to query
MongoDB each time
"""
def create_persistent_dictionaries():
    keyword_regex, scenario_keywords = read_regex_mongo()
    keyword_dict_compiled_pattern = create_keyword_dictionary(keyword_regex)
    scenario_dictionary = create_scenario_dictionary(scenario_keywords)
    return keyword_dict_compiled_pattern,scenario_dictionary

"""
Actual method for tagging a string.
Note that it accepts the keyword dictionary (with compiled patteern) and scenario dictionary (with keyword sets)
to separate query of MongoDB from actual tagging
"""
def tag_input_string(input_string,keyword_dict_compiled_pattern, scenario_dictionary):
    tagged_keywords = regex_tag(input_string, keyword_dict_compiled_pattern)
    tagged_scenario = tag_scenario(tagged_keywords, scenario_dictionary)
    return tagged_scenario

"""
Example of what should be in main, comment out once the rest of the code has been settled
Note: Please put creation of persistent dictionary outside of any tagging method!!
"""
# keyword_dict_compiled_pattern,scenario_dictionary = create_persistent_dictionaries()
# tagged_scenario = tag_input_string("why bar", keyword_dict_compiled_pattern, scenario_dictionary)
# print(tagged_scenario)

