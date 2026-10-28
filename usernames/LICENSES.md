# Data sources and licences

| File | Source | Licence |
| --- | --- | --- |
| `terms/ldnoobw_en.txt` | [List of Dirty, Naughty, Obscene, and Otherwise Bad Words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), file `en`, commit `5faf2ba42d7b1c0977169ec3611df25a3c08eb13` (2020-07-13), unmodified | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: © Shutterstock, Inc. and the LDNOOBW contributors. |
| `terms/caribbean_tt.txt` | Written for All Fours; approved by the owner 2026-10-28 | This repository's licence |
| `terms/exclusions.txt` | Written for All Fours | This repository's licence |
| `words/web2_sample.txt` | A deterministic sample of `/usr/share/dict/words` (Webster's Second International, 1934, copyright lapsed; as shipped by FreeBSD and macOS) | Public domain |
| `words/propernames.txt` | `/usr/share/dict/propernames` (FreeBSD and macOS) | Public domain |
| `words/handle_words.txt`, `eval/*.txt` | Written for All Fours | This repository's licence |
| `dataset/*.tsv` | Generated from the files above by `build_dataset.py` (seeded, reproducible) | Inherits the above; the LDNOOBW-derived rows are CC BY 4.0 |

The base model is [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert), revision
`4d6c22e74ba2fdd26bc4f7238f50766b045a0d94`, Apache-2.0 (Unitary A.I. Ltd., "Detoxify", 2020). The fine-tuned weights
are a derivative under the same licence and are not committed (see README.md). This repository is a fork of [unitaryai/detoxify](https://github.com/unitaryai/detoxify) (Apache-2.0).
