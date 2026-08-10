# Data, privacy, and intended-use statement

## Contents and intended use

The release contains derived, row-level diagnostic scores and aggregate summaries for research auditing and reproduction of the reported tables. It is not intended to train a language model, profile individuals, or support production requirements, simplification, or grammatical-correction decisions.

## Source and transformation

Source datasets are listed in `THIRD_PARTY_ASSETS.md`. The released SRS pair table retains collision-safe experimental identifiers, dataset labels, operation labels, technical-availability flags, and numeric scores. Full source inputs, perturbations, rewrites, references, and judge responses are omitted.

## Identifiers and privacy

The archive was checked for author names, affiliations, institutional domains, personal accounts, email addresses, persistent learner identifiers, absolute home paths, credentials, and repository history. None are included. W&I+LOCNESS learner IDs and personal-looking content, and contact details occurring in public requirements documents, are absent because all raw text is excluded.

## Offensive or sensitive content

No natural-language corpus text is distributed, which removes the principal path by which offensive, identifying, or sensitive content could be exposed. The remaining model/dataset names and numeric diagnostics are not person-level records.

## Known limitations

The data cannot independently regenerate model rewrites or text-level pair scores without separately acquired upstream datasets and model executions. This is an intentional licensing and privacy boundary. The released score rows, hashes, frozen contract, and executable aggregator support exact auditing of the reported SRS aggregates.

