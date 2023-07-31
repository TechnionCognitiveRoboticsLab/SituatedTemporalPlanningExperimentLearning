#!/bin/sh

for p in pddl-instances/rcll-fixed-dl/problem*.pddl; do
    echo "./rewrite-no-lp --search-dump $p.dump.txt --include-metareasoning-time --multiply-TILs-by 1 --calculate-Q-interval 100 --add-weighted-f-value-to-Q -0.000001 --min-probability-failure 0.001 --slack-from-heuristic --forbid-self-overlapping-actions --deadline-aware-open-list IJCAI --ijcai-gamma 1 --ijcai-t_u 100 --icaps-for-n-expansions 100 pddl-instances/rcll-fixed-dl/rcll_domain_production_durations_time_windows.pddl $p > $p.log";
done