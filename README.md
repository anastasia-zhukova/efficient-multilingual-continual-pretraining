# Efficient domain-adapted continual pretraining for (almost) any language

## How to run

1. Install the dependencies with `poetry`;
2. Select the required task in `config.yaml`;
3. Make the `model_weights` folder in root if not present;
4. Run `python3 main.py`.

## Project overview

### Datasets

In this work we've used 8 different datasets:

- Amazon reviews
- OpenRepairData
- ChemProt
- RCT
- CARES
- NUBes
- Pharmaconer
- Cantemist

The different datasets correspond to different domains and languages. Amazon reviews and OpenRepairData both stand for
German language and Electronics domain; ChemProt and RCT datasets are both for English language and BioMed domain;
NuBes, Pharmaconer, Cantemist and CARES are all four for Spanish + BioMed domain. This would allow us in the future
to test our approach in different domain+language combinations.

Below we provide a brief overview of the datasets and the tasks.

#### Amazon reviews

The goal is to predict the rating a user gave to a product based on their comment. This is a multiclass classification
task with 5 classes (one for each possible rating). The dataset, unfortunately, is no longer available publicly.

The dataset contains reviews on products in different categories. In order to keep it to a single domain (electronics),
we've decided to only use the comments from "Electronics" category written in German. While the dataset provides more
metadata than just the comment text, we decided to only use it to not mix up additional effect from other features to
our work.

#### OpenRepairData

The dataset includes 2 main columns: the problem description and the solution to it. The goal for the model is to match
the problems to their solutions. In order to do this, we transformed the task to a binary classification task: all the
possible problem-solution pairs are generated, which are then passed to the model. The model predicts the probability
that the given solution corresponds to the given problem.

The dataset provides the "positive" problem-solution pairs. In order to train the model, we also need "negative" ones
(the pairs which do not exist in reality). To do this, we simply randomly generate the pairs from the common pool,
excluding the ones which are present in reality. The ratio of positive to negative pairs is a variable that affects
both the training time and the training quality; for our tests, we used `1:5`.

#### ChemProt

The dataset consists of articles' abstracts, in which chemical and biological entities positions are given. The goal
is to predict the relationship between the given entities. Since this is a complex task which in fact requires multiple
steps, we decided to narrow it down to a following scenario: given the positions of two entities in text, predict
their relationship. That means that we do not find these entities and do not filter them. Otherwise, we would also have
to perform the NER task here. As in the original task, we only used the relations present in so-called `gold_standard`
file (it has some limitations to which relations are actually taken into account). Finally, the title is also present
in the data and is simply added to the original text which is then provided to the model.

As of the task, given we know the positions of the entities, the task is a multiclass classification.

The dataset appeared to be highly imbalanced with one of the 5 classes being present roughly 10 times more often than
the other one. To solve this, a weighted loss was used with weights inversely proportional to the sizes of the classes.

#### RCT

This dataset also includes the abstracts' of papers, but this time we have to predict the sentence sentiment role
of the sentences present in it. There are 5 possible classes, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS, BACKGROUND.
This task is quite similar to the previous one,
given that there is only a single entity we need to make a prediction for (sentence), so the approach is also similar.
For this task, we used a smaller version of dataset to correspond to the "low-resource" scenario with digits
being present.

#### NUBes

The dataset provides an insight on how negation and uncertainty is expressed in speech. It provides sentences with
positions of negation/uncertainty markers as well as what is actually being affected by those. While it provides a lot
of different classes for the affected content (like Disorders, Procedures and various scopes), we decided to simplify the task by introducing a single 
category for all the **affected** content. This seems like to be a logical simplification, since in the end we want to
know **what** is being affected and **how**.

This is a classic NER scenario task where you have to find entities of different classes.

#### PharmaCoNER

This is a corpus, which consists of documented clinical cases. For each case description, the chemical and protein
entities positions are provided. Here, we had to solve another NER task with 4 entity types.

#### CANTEMIST

The dataset provides oncological clinical cases reports. Similarly to PharmaCoNER, it contains annotations for entities,
this time it is the tumor morphology ones (still a NER task).

#### CARES

This is a corpus of anonymised radiological evidences (in Spanish), or CARE(S). It provides the descriptions of
evidences and the ICD-10 codes which are connected to each of the evidences. While the dataset provides a detailed
and complete ICD-10 classification, in order to reduce the number of classes, we decided to only use the general
ICD code (column `general` in the dataset). Since the description can be associated with more than one code, this is a
multi-label task we need to solve.

### Continual pretraining of models

To further pretrain the models, we adopt the techniques
from [Don't stop pretraining](https://aclanthology.org/2020.acl-main.740/),
namely DAPT and TAPT. For both modes, we continued pretraining the already pretrained model on the corresponding
dataset on the **masked language modelling task only**.

For this task, we adopt another way to count how long our model is being trained.
Instead of using epochs, we stick to the "steps" notation with a single step being one optimizer step done.
In case the dataloader is exhausted, it is simply reset. The logging is done every `n` epochs in a way to have
around 20 points to plot the graphic.

#### DAPT setup

For DAPT, we used and compared two approaches: using a "manually" collected DAPT and the one collected
using [DSIR](https://arxiv.org/abs/2302.03169). For DSIR, we used the data from all the downstream tasks
to collect the dataset. Then, the model was being continually pretrained for 85000 steps with batch size 8.

The manually collected dataset included 1.2M entries, while the DSIR-collected one was only 100k.

We tried using different learning rates for this task and ended up using 1e-4 as the best-performing one in terms
of validation loss on MLM task.

#### TAPT setup

For TAPT, we used the data from all the tasks mixed together. The general approach was the same as the one we used for
DAPT, but since there is much less data, we ended up using 40000 steps with batch size 8.

### Downstream tasks methodology

#### Amazon data

For this task, only the comment text is used from the dataset as noted above. After being tokenized, the comment texts
are passed to the model. First, the model core produces the embeddings. Then, we use the `CLS` embedding as the one
supposedly containing all the necessary information from the sentence and pass it through the following head:

```python
nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.Linear(hidden_size // 2, num_labels),
)
```

which gives us logits for each of the classes. We use the default `CrossEntropyLoss` for loss and `AdamW` with
parameters from the config to train the model. These are used for all the models if not listed otherwise.

`deepset/gbert-base` is used as a core model.

#### OpenRepairData

The common approach to the task has been already listed above in the `Datasets` section.
The negative pairs generation is done using the pipeline's `_generate_data` method. Since there are no train/val sets,
it is done with a train test split by the same method. After we have the problem and the solution texts, embeddings for
each one are built independently. As in the previous task, we extract the `CLS` token's
embeddings. We concatenate the two embeddings (for problem and solution) and pass it through the head:

```python
nn.Sequential(
    nn.Linear(2 * hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.Linear(hidden_size // 2, num_labels),
)
```

`num_labels` here is equal to 2 for common model interface.

`deepset/gbert-base` is used as a core model.

#### ChemProt

In this task, loading data is one of the tricky parts. It includes several steps:

1. The entities with their locations in text are acquired from `enitities.tsv` files;
2. The relevant relations between these entities are determined from `gold_standard.tsv` files;
3. The texts themselves are obtained from `abstracts.tsv`. Only the texts with the relations are included, the rest
   could be used in the future for language modelling tasks pretraining. As noted above, we concatenate the name and the
   abstract body together.

The collation function is of particular interest here. Since the abstracts are long, not all of them can fit in the
token limit of the model. To simplify the task, we simply truncate the abstracts and ignore all the entities which were
in the truncated part. We then figure out the indexes of the tokens which the entities' mentions in the text were split
to and pass those to the model along with the tokenized input.

The model's core then generates embeddings for all tokens. Then, embeddings of the tokens which are the part of
the entities are averaged. Since we have two entities in a relation, we thus obtain 2 embeddings which are then
concatenated just like we did in the previous task. They are then passed through the head to get the predictions. The
head for this task is as follows:

```python
nn.Sequential(
    nn.Linear(2 * hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.Linear(hidden_size // 2, num_labels),
)
```

Since for this task the data is highly imbalanced, we use weighted `CrossEntropyLoss` as noted above.

`FacebookAI/roberta-base` is used as a core model.

#### RCT

This task is quite similar to ChemProt and the methodology here is the same except we only have one "entity" (sentence)
instead of two. That means we no longer need the concatenation in the last step so the head is a little different:

```python
nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(hidden_size // 2, num_labels),
)
```

For this task, there is no need to use the weighted loss, so the setup is default.

`FacebookAI/roberta-base` is used as a core model.

#### CARES

The pipeline here resembles the pipeline for Amazon data, with the difference being that the model makes prediction
for all the possible labels and then uses sigmoid instead of softmax since each label prediction is now independent of
others. The process to construct targets matrix is also a bit different from the one we would see in binary/multiclass
classification: instead of providing a 1D tensor with classes, we instead provide a 2D tensor with relevant classes
having `1` and non-relevant having `0` in the matrix.

For loss, we use the `BCEWithLogitsLoss` with default parameters. The head looks as follows:

```python
nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.Linear(hidden_size // 2, total_classes),
)
```

`IIC/BETO_Galen` is used as a core model.

#### NER tasks

Since all the NER tasks (CANTEMIST, PharmaCoNER, NUBes) share the same methodology, the main difference being the data
itself, it makes sense to group them.

For all the datasets, we adopt the common annotation pattern: the tokens which start the entity are marked as
`B-entity_name`, the ones that "continue" it are marked as `I-entity_name`. The annotations are stored in `.ann` files
in `annotations` folder and the texts are stored in `.txt` files in `texts` folder inside the train/val folder.
The annotations should follow the BRAT format.

All the tokens which are not a part of an entity are marked as class 0. After that, the task is essentially a
multiclass classification task where we need to classify each token.

The NUBes data has a little different input data format since we need to further group the affected parts of the
sentence as noted above which is the reason for it to have a slightly different dataset and thus pipeline.

The head for all the task is included in the model class and is as follows:

```python
nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(self.bert.config.hidden_size // 2, self.bert.config.hidden_size // 4),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(self.bert.config.hidden_size // 4, n_classes),
)
```

`IIC/BETO_Galen` is used as a core model.

### Evaluation

Depending on task, we used different approaches and tools to evaluate the model:

- Binary classification: default set of metrics: accuracy, precision, recall, f1-score
- Multiclass classification: same metrics list as above, but we use the macroaveraging to calculate the final metrics.
- Multi-label classification: we assume that everything the model predicted more than 0.5 is "label is present" and
  then it is the same calculation as the one for multiclass.
- NER: we use the f1-score, precision and recall from the `seqeval` in `strict` mode.

The first three calculations are done using the `MetricCalculator`. It is built to be able to work in any of the given
modes to make the train/validation loop more simple. The more complex "multilabel" and "multiclass" modes are covered
with unit tests to ensure correct calculations. The NER calculations are done with a similar `NERMetricCalculator` to
which wraps the `seqeval` calculations to ensure the common interface.

### Notable features

- The project uses `loguru` to log some details of the process to make it easier to debug the project and get as-you-go
  information. The logger setup can be found in `efficient_multilingual_continual_pretraining/logger.py`.
- We use `wandb` to monitor the models and store the future results to make experiments easier to track. It is turned
  on and off by the `use_watcher` in `config.yaml`. Make sure to login before you run it.
- Hydra is used to manage configs in a simple way. By selecting the required task in `task` (same as task's yaml name),
  it allows to easily change the task we are training and keep the hyperparameters different for different tasks.
  It also allows for command line overrides if necessary.
- `poetry` is used to manage the project dependencies and to remove the unnecessary development dependencies in the
  future production.
- The tasks are run from a single `main.py`. After the run, the models are being saved to `model_weights` folder.

### Preliminary results

#### Downstream tasks: pretrained model (train scores)

We provide these as a reference for future researchers only.

| Domain               | Model used   | Dataset        | Task type                   | Accuracy | Precision | Recall | F1 Score |
|----------------------|--------------|----------------|-----------------------------|----------|-----------|--------|----------|
| German + Electronics | gbert-base   | Amazon reviews | Multiclass classification   | 0.879    | 0.697     | 0.696  | 0.696    |
| German + Electronics | gbert-base   | OpenRepairData | Binary classification       | 0.762    | 0.385     | 0.717  | 0.501    |
| English + BioMed     | roberta-base | ChemProt       | Relationship classification | 0.823    | 0.778     | 0.231  | 0.426    |
| English + BioMed     | roberta-base | PubMed 20k RCT | Sentences sentiment roles   | 0.946    | 0.814     | 0.799  | 0.804    |
| Spanish + BioMed     | BETO_Galen   | CARES          | Multilabel classification   | 0.999    | 0.960     | 0.900  | 0.923    |
| Spanish + BioMed     | BETO_Galen   | CANTEMIST      | NER                         | N/A      | 0.219     | 0.191  | 0.204    |
| Spanish + BioMed     | BETO_Galen   | PharmaCoNER    | NER                         | N/A      | 0.542     | 0.514  | 0.528    |
| Spanish + BioMed     | BETO_Galen   | NUBes          | NER                         | N/A      | 0.615     | 0.344  | 0.441    |

#### Downstream tasks: pretrained model (validation scores)

| Domain               | Model used   | Dataset        | Task type                   | Accuracy | Precision | Recall | F1 Score |
|----------------------|--------------|----------------|-----------------------------|----------|-----------|--------|----------|
| German + Electronics | gbert-base   | Amazon reviews | Multiclass classification   | 0.877    | 0.700     | 0.696  | 0.698    |
| German + Electronics | gbert-base   | OpenRepairData | Binary classification       | 0.703    | 0.288     | 0.531  | 0.373    |
| English + BioMed     | roberta-base | ChemProt       | Relationship classification | 0.781    | 0.763     | 0.201  | 0.402    |
| English + BioMed     | roberta-base | PubMed 20k RCT | Sentences sentiment roles   | 0.946    | 0.815     | 0.807  | 0.806    |
| Spanish + BioMed     | BETO_Galen   | CARES          | Multilabel classification   | 0.990    | 0.289     | 0.227  | 0.236    |
| Spanish + BioMed     | BETO_Galen   | CANTEMIST      | NER                         | N/A      | 0.242     | 0.232  | 0.237    |
| Spanish + BioMed     | BETO_Galen   | PharmaCoNER    | NER                         | N/A      | 0.242     | 0.272  | 0.257    |
| Spanish + BioMed     | BETO_Galen   | NUBes          | NER                         | N/A      | 0.675     | 0.450  | 0.540    |

#### Downstream tasks: comparison with augmented models (validation scores)

| Dataset        | Enhancement   | Accuracy | Precision | Recall | F1 Score |
|----------------|---------------|----------|-----------|--------|----------|
| Amazon reviews | None          | 0.877    | 0.700     | 0.696  | 0.698    |
| Amazon reviews | DAPT (manual) | 0.861    | 0.654     | 0.658  | 0.653    |
| Amazon reviews | DAPT (DSIR)   | 0.783    | 0.697     | 0.684  | 0.679    |
| Amazon reviews | TAPT          | 0.882    | 0.705     | 0.709  | 0.701    |
| OpenRepairData | None          | 0.703    | 0.288     | 0.531  | 0.373    |
| OpenRepairData | DAPT (manual) | 0.790    | 0.326     | 0.240  | 0.276    |
| OpenRepairData | DAPT (DSIR)   | 0.702    | 0.287     | 0.528  | 0.372    |
| OpenRepairData | TAPT          | 0.782    | 0.340     | 0.329  | 0.334    |

The results for the remaining tasks are to be announced. 

## Discussion

First of all, unfortunately, at the given moment it is not possible to claim that DAPT improves the quality over the
base pretrained models. Though using TAPT might result in slight improvement, the latter is not drastic. 

We've experimented with using different learning rates for continual pretraining of the model. It has been shown that
the 1e-5 learning rate significantly outperforms both 1e-6 and 1e-4 in terms of loss (train, validation and test
-- all at once). We've tried changing the random seed, but the result remained consistent, so we decided to stick to it.

Finally, it seems like there is not real difference between using manually collected DAPT and DSIR DAPT. Since DSIR
DAPT collection can be automated, it may be promising once we find circumstances when DAPT provides a performance
increase.

---

## Development

**tldr**: use your branch, tests and logging are good, NumPy style for docstrings, use poetry, isort, black and ruff.
Name stuff adequately, use PEP and all the programming knowledge you have.

---

### Python

Most packages are stable at python version 3.10, so we use it as default as noted in
`pyproject.toml`. If you have any reasoning against that, please let us know and we'd discuss that.


- Each person has their own branch for development, please use those rather than pushing everything to `master`

- Naming of the branches should be so that it would be possible to identify
  who is the owner of a branch (for example, `jd-dev` where "jd" stands for John Doe).

- Avoid using `--amend` and `force` stuff (only use when necessary) as those might affect the work of other people.
- Write readable commit messages.
- In general, the `master` branch should only contain code that is stable and "production-ready"
(so no random exceptions would show up if we were to run it).
- Avoid storing and versioning some redundant files like `123123.py` or `test.ipynb`. Only stage and version
  the stuff that is relevant to the project.

- Data goes to `data` (make dirs inside for different data sources, for example), notebooks go to `notebooks`,
  development code goes to the `efficient_multilingual_continual_pretraining` source.

### Tests

Tests are good (we suggest using `pytest`) but not necessary and should be kept near the code being tested.
For example, you could have:

```
.
└── efficient_multilingual_continual_pretraining/
    └── models/
        ├── tests/
        │   └── test_my_model_1.py
        ├── my_model_1.py
        └── __init__.py
```

### Logging

Logging is also good and is often helpful. `loguru` provides straightforward logging out-of-the-box.
Please store logs in the `logs` folder which is excluded from versioning in `.gitignore`
and do not version those to avoid clutter.

### Poetry

We use [`poetry` to manage the dependencies](https://python-poetry.org/). The stuff you should know:

- `poetry install` installs the current dependencies (including `dev` ones).
- `poetry add <package_name>` works similarly to `pip install`.
- The installed packages can be seen in `pyproject.toml`.
- `poetry` also generates `poetry.lock` file. Be sure to version it as well.

### Codestyle

We also suggest using [**File watchers
**](https://medium.com/compendium/automatically-run-black-in-pycharm-on-windows-d2eab855a918)
to keep consistent codestyle. The recommended watchers are `isort`, `black` and `ruff`. You could use the `watchers.xml`
found
[here](https://drive.google.com/file/d/1ycj9xTUWl4bfDnEbvlBcunvW8QBcbjvX/view?usp=sharing).
Replace the paths to programs to the ones you have.

Talking about docstrings, we suggest using
the [NumPy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

#### Known watcher issues

1. `isort` aims to clear the unused imports. If you are making an import in the
   `init.py` file, the import most probably will be unused. To solve this, use the
   `__all__` [(stackoverflow)](https://stackoverflow.com/questions/44834/what-does-all-mean-in-python):
