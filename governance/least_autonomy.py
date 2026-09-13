"""
governance/least_autonomy.py - least-autonmy enforcement checks.
verifies that check_support_ticket_status is only available to lookup_agent.
"""

from __future__ import annotations

from crewai import Agent


EXPLANATION = """
Least-Autonomy Enforcement — Application Layer (Task 15)
-------------------------------------------------------
Only lookup_agent has access to check_support_ticket_status.
retrieval_agent has rag_lookup only, and composer_agent has no tools.
CrewAI dispatches tools strictly from the agent's tool list, preventing
other agents from running unassigned tools.
"""


def assert_least_autonomy(
    ret_agent: Agent,
    lkp_agent: Agent,
    cmp_agent: Agent,
) -> None:
    """checks that ticket status tool is wired only to lookup agent"""
    from tools.ticket_tool import check_support_ticket_status

    lookup_tool_names = {t.name for t in lkp_agent.tools}
    retrieval_tool_names = {t.name for t in ret_agent.tools}
    composer_tool_names = {t.name for t in cmp_agent.tools}

    target_name = check_support_ticket_status.name

    assert target_name in lookup_tool_names, (
        f"FAIL: '{target_name}' not found in lookup_agent.tools!"
    )
    assert target_name not in retrieval_tool_names, (
        f"FAIL: '{target_name}' is in retrieval_agent.tools — least-autonomy violated!"
    )
    assert target_name not in composer_tool_names, (
        f"FAIL: '{target_name}' is in composer_agent.tools — least-autonomy violated!"
    )


def demonstrate_least_autonomy() -> None:
    """prints tool mapping to verify isolation between agents"""
    from crew.agents import retrieval_agent, lookup_agent, composer_agent

    print("\n" + "=" * 60)
    print("  Least-Autonomy Enforcement Verification")
    print("=" * 60)

    agents = {
        "retrieval_agent": retrieval_agent,
        "lookup_agent":    lookup_agent,
        "composer_agent":  composer_agent,
    }
    for name, agent in agents.items():
        tool_names = [t.name for t in agent.tools]
        print(f"\n  {name}:")
        print(f"    tools = {tool_names}")

    assert_least_autonomy(retrieval_agent, lookup_agent, composer_agent)
    print("\n  ✓ check_support_ticket_status is wired to lookup_agent ONLY.")
    print("  ✓ retrieval_agent and composer_agent do NOT hold the ticket tool.")
    print("=" * 60 + "\n")
    print(EXPLANATION)


if __name__ == "__main__":
    demonstrate_least_autonomy()
