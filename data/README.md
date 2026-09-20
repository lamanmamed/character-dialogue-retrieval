# Dataset

The original dialogue CSV files are not redistributed in this repository.

The evaluation code expects tab-separated files containing at least these columns:

- `Character_name`
- `Line`

The source coursework data also contained `Episode`, `Scene`, `Scene_info`, and `Gender`, but the retrieval pipeline only needs the character name and spoken line.

The original experiment used at most the first 300 non-empty lines for each character in the training file and at most the first 50 non-empty lines for each character in validation and test files.
