# === CELL 10: smoke-test the REAL agent — TEST ONLY, safe to leave in a submission notebook ===
# Reuses the `agent` that YOUR main.py CELL defined (main.py did cg import + model load + config).
# It drives ONE isolated cg battle (MCTS vs random); env.run() would be in-process and collide
# with the MCTS cg-search.
#
# GUARD: if `agent` is not defined this cell SKIPS. In a SUBMISSION run main.py is %%writefile'd
# (written to disk, NOT executed) so `agent` is undefined -> this skips cleanly and never breaks
# the commit. To actually run the test: interactively RUN a main.py cell (not %%writefile) first.
#
# Order to TEST:  Restart kernel -> cell 07 -> RUN main.py as a cell -> this cell.
exec(open("/kaggle/working/main.py").read())
print("agent ready:", "agent" in dir())
if "agent" not in globals():
    print("`agent` not defined (main.py was written, not run) -> skipping smoke test.")
else:
    import random
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class
    import nn_agent
    DECK = list(nn_agent.MY_DECK)

    def _rnd(o):                         # random legal move (our own deck)
        oo = to_observation_class(o)
        if oo.select is None:
            return DECK
        a = nn_agent.enumerate_actions(oo)
        return random.choice(a) if a else []

    print("playing: main.py's agent (MCTS) vs random ...", flush=True)
    random.seed(0)
    obs, _ = battle_start(DECK, DECK)    # main.py's agent = seat 0
    turn = 0
    while obs["current"]["result"] < 0:
        st = obs["current"]
        sel = agent(obs) if st["yourIndex"] == 0 else _rnd(obs)
        obs = battle_select(sel)
        turn = max(turn, st.get("turn", 0))
    battle_finish()
    r = obs["current"]["result"]
    print("RESULT:", {0: "seat0 (us) WIN", 1: "seat1 win", 2: "TIE"}.get(r, r), "| turns:", turn)
    import nn_agent_mcts
    st = nn_agent_mcts.gate_stats()
    print("stats:", st)
    print("opp-model engaged:", "YES" if st.get("opp_hit", 0) > 0 else "NO (mirror should match!)",
          "| advisor engaged:", "YES" if st.get("adv_hit", 0) > 0 else "NO",
          "| adv errors:", st.get("adv_err", 0))
    print("No error -> the REAL main.py agent (opp-model + advisors) plays in Kaggle's cg env.")
