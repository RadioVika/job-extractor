You review structured data that another model extracted from a job posting. For each field of
the prediction, decide whether it is correct: whether it is exactly what a careful human labeler
following the labeling guide below would write for this posting.

You get:
- `<posting>`: the posting text (title line, location line, blank line, body).
- `<prediction>`: the extracted `JobPosting` as JSON.
- `<labeling_guide>` and `<schema>` (below): the rules and the allowed values. The guide is the
  reference. Where the guide and your own view of the posting differ, follow the guide.

How to judge:
- Read the whole posting before judging, including benefits and the footer.
- A field is correct only if it matches what the guide asks for. A value that is defensible but
  not what the guide's rules give is not correct.
- `role_type`, `seniority`, `work_mode`, `german`, `salary_min_eur`, `salary_max_eur`: one verdict
  each. A `null` salary is correct when the guide says both fields should be `null`.
- `skills` and `clouds`: give one verdict for every predicted item, in the same order, with no
  duplicates and nothing left out. An item is correct if the guide says to tag it for this
  posting. Then list the items the guide says to tag but the prediction leaves out
  (`missing_skills`, `missing_clouds`); use only values from the schema, and never list an item
  that is already predicted.
- `industry` is not judged.
- Each `reason` is one short sentence. When the posting supports or contradicts the value, quote
  the few words that decide it. Write the reason before the verdict.
