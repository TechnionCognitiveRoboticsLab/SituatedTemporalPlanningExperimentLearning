#!/bin/sh

echo search_algorithm, heuristic, domain, problem, expansions, solved, search_dump_file


for p in pddl-instances/rcll-fixed-dl/problem*.pddl; do
    SOLVED=`grep -c ';;;; Solution Found' $p.log`
    EXPANSION_LINES=`wc -l $p.dump.txt | cut -f 1 -d ' '`
    EXPANSIONS=`expr $EXPANSION_LINES - 1`
    DOMAIN=RCLL
    PROBLEM=$p
    SEARCH_ALGO=SITUATED
    HEURISTIC=TFF

    gzip $p.dump.txt

    echo $SEARCH_ALGO, $HEURISTIC, $DOMAIN, $PROBLEM, $EXPANSIONS, $SOLVED, $p.dump.txt.gz
done