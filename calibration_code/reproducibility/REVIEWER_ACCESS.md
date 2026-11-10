# Editor/reviewer access record

## Public component

- Repository: <https://github.com/tanbinyu/clinical-triage-lec>
- Release and exact commit: read from `manuscript_run_manifest.json`; the
  manifest must report `matches_declared_release: true` for a tagged release.
- Public numerical source data: the journal Source Data workbook submitted
  with the article.

## Controlled component

The following de-identified materials should be uploaded as a reviewer-only
file in the Nature submission system, or shared through a time-limited
institutional link entered in the confidential editor note:

1. the completed `Reviewer_Reproducibility_Bundle.zip`;
2. the frozen routed prediction JSON exports, only where redistribution terms
   permit;
3. trained adapter files and checkpoint acquisition records, only where the
   underlying model licences permit redistribution.

If an input cannot be redistributed, provide its official acquisition URL,
version/checksum and a script that validates the locally acquired copy. Never
upload source medical images or direct identifiers in this bundle.

The access route is not complete until the submission-system upload ID or the
controlled-link location, expiry and access contact have been entered in the
confidential submission record. This repository deliberately does not invent
such an ID before the upload exists.
