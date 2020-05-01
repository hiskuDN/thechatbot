# import the github dataset and fill data into the intents.json file
import json


def prep_entity_json():
    # import languages.json and export all languages to entities.json
    # Already done
    print('preping entiyties.json file')
    with open('languages.json').read() as data_file:
        langList = json.loads(data_file)

    with open('entities.json').read() as data_file:
        entities = json.loads(data_file)

    for lang in langList['languages']:
        entities['entities'].append({
            "value": lang,
            "synonyms": []
        })

    with open('entities.json', 'w') as outfile:
        json.dump(entities, outfile)


def prep_intent_json():
    print('preping intents.json file')
