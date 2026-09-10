# Prompt for the New Codex Account

Paste the following into the first task in the new Codex account after opening
the live project checkout.

```text
Continue development of Cognitive-Inspired-View-Selection from the existing
working tree. This task is for coding, experiment design, execution support,
and figure-generation infrastructure. A separate ChatGPT handoff is the
authoritative source for paper prose and the current manuscript argument.

Repository on this Mac:
/Users/hankkung/Documents/Documents - Hank’s MacBook Pro/GitHub/Cognitive-Inspired-View-Selection

Repository on the compute server:
/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection

Start by reading CODEX_HANDOFF.md, MVSelect-main/README.md, and the README for
the relevant analysis. Then run git status --short in the live Mac checkout.
The working tree contains a large, valuable, uncommitted implementation that is
not on origin/main. Do not run git reset, git clean, checkout over files, pull
through conflicts, or revert changes. Work with the existing state and preserve
all generated experiment data.

Important conventions:
- Summarize every file modified after each edit.
- Use expanded family and foreshortened family; do not call them canonical.
- State whether an analysis uses the initial scout, selected actions only, or
  the full classifier input.
- State whether it uses the final classifier or classifier-at-epoch-t.
- Keep freeze sweeps and selector-limit sweeps in separate figures.
- Keep architecture outputs separate for resnet18, vit, and tinyvit.
- Do not add unrequested metrics or insider wording to public documentation.

The most recent experiment is an active-single control. With
--active_single_view and --steps 1, the selector observes one random scout,
selects one action view, and the classifier receives only that selected view
(N=1). The scout is only a detached reward baseline. Standard view-ratio,
family-ratio, lift, and N=1 accuracy plots are valid. Pairwise confidence,
leave-one-out, selected-N=5 analyses, and selection-JSON temporal merging are
not valid active-single analyses without redesign. See the compatibility matrix
in CODEX_HANDOFF.md before adapting any analysis.

Before making changes, report:
1. the branch and concise working-tree status;
2. whether the requested script is compatible with the intended N and mask
   semantics;
3. the files you expect to touch;
4. the smallest valid smoke test available on the server.

Do not restart the project from origin/main. Continue from the live files.
```

