# The Chat Bot Project

## Notes on the Project

* *intents.json* should contain training data to create the model. It should be split into two classes which are the training data and the test data.

__Suggestion Here: In our case, since we have a different dataset from the intent.json, we may need to import and convert the dataset to a type that we want to__

* *intents.json* is also used to generate responses based on the generated classificatoin from our model

* Use *botTraining.py* to train the model from *intents.json*

* The model we generate will classify what the user's intents are and responds with a unique "tag" that just like defined inside the intents file

* After our model classifies the user input into one of the intent tags, we iterate through the json file to find a response.