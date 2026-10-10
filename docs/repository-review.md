# Review of the related portfolio repositories

Reviewed the published main-branch trees and README files of [green-taxi-ray-project](https://github.com/shlomi1993/green-taxi-ray-project) and [green-taxi-mlflow-project](https://github.com/shlomi1993/green-taxi-mlflow-project) while preparing this repository. This is a layout and documentation review, not a full correctness audit.

## Existing strengths

The Ray project separates preparation, scoring, actor state, replay strategies, and artifacts under `src/`; includes Docker cluster setup, tests, a demo, and checked-in output examples. Its README explains skew, timeouts, bounded concurrency, and fallback semantics.

The MLflow project separates Metaflow orchestration from pipeline services; places gate settings in `configs/`; and includes architecture, design, and operations documents under `docs/`. Its README clearly explains champion/candidate evaluation and model promotion.

This project follows those conventions with root entry points, reusable modules, tests, separate documentation, and compact output examples.

## Suggested improvements

| Repository | Change | Benefit |
|---|---|---|
| Both | Add GitHub Actions checks and link the actual workflow status | Makes validation visible to reviewers; neither inspected tree includes `.github/workflows` |
| Both | Choose an explicit license if public reuse is intended | Neither inspected tree includes a license file |
| Both | Put a concise result table and reproducible command near the top | Helps a CV reviewer see evidence quickly |
| Both | Keep diagrams in the repo with editable source | Makes visuals maintainable; some README figures use user-attachment URLs |
| Ray | Move the design document into `docs/` and shorten the main README | Makes the entry page easier to scan |
| Ray | Document editable installation before using packaged commands | README uses console commands while setup only creates the Conda environment |
| MLflow | Reconcile strict promotion inequalities in prose | Flow steps say greater than 1% and less than 5%; governance text says at least 1% and no more than 5% |
| Both | Explain the choice to version sample Parquet files alongside download scripts | Removes uncertainty about which data is needed for the demo |

No changes were made to either existing repository. The new repository includes CI, editable installation, Mermaid diagrams, and honest example provenance. A license is left for the owner to choose rather than assigning reuse rights during extraction.
