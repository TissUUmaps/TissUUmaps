---
"@tissuumaps/storage": minor
"tissuumaps": minor
---

AnnData tables name the features of `X`, the layers and `raw` by a `var` column (`featureNameColumn`, chosen by convention if omitted), e.g. gene symbols instead of Ensembl IDs; names shared by several features are qualified by their `var` index.
