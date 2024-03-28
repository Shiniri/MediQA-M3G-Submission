
# Description

This code contains our submission for the MediQA challenge 2024, 2nd task:
Multilingual & Multimodal Medical Answer Generation. It converts the training
data to the format used by the LLaVA-Med model, runs training & inference
and then converts predictions back to the challenge required format.

**Note**:
Generation is not fully deterministic, so results might differ slightly
from the final_prediction.json, which represents our submission.


# Running the experiment

1. Clone this repository *recursively* to include LLaVA-Med.
```
git clone --recursive git@github.com:Shiniri/MediQA.git
```

2. Include a valid Llama-7b checkpoint in the repository,
   as well as: the images for the training data in `./data/images_train`
   and the images for the test data in `./data/images_test`.
   Also include the training and test json files from the challenge in
   `./data/test.json` / `./data/train.json`.

3. Follow the setup instructions of the original LLaVA-Med repository [here](https://github.com/microsoft/LLaVA-Med).

4. Set the Llama path variable in the `./run_experiment` script to
   point to your Llama checkpoint and execute it.
   **Note:** you can probably leave out certain parts of the script depending on whether
   you want to re-run data conversion, training, etc.