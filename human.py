import json
from collections.abc import Callable
from typing import Any


Observation = dict[str, Any]
Request = dict[str, Any]


def validate_deck(deck: list[int]) -> None:
    if len(deck) != 60:
        raise ValueError(
            f"Deck must contain exactly 60 card IDs, but got {len(deck)}."
        )

    if not all(isinstance(card_id, int) for card_id in deck):
        raise TypeError("Every card ID must be an integer.")


def collect_options(obs_dict: Observation) -> Request:
    select = obs_dict.get("select")

    if select is None:
        raise RuntimeError("A deck request does not contain action options.")

    options = select.get("option") or []
    min_count = select.get("minCount", 0)
    max_count = select.get("maxCount", 0)

    if not options and max_count > 0:
        raise RuntimeError("CABT requires a selection but returned no options.")

    return {
        "type": select.get("type"),
        "context": select.get("context"),
        "min_count": min_count,
        "max_count": max_count,
        "options": options,
    }


def display_options(request: Request, decision_number: int) -> None:
    print("\n" + "=" * 70)
    print(f"Decision #{decision_number}")
    print(f"select type: {request['type']}")
    print(f"context: {request['context']}")
    print(
        "required selections: "
        f"{request['min_count']} to {request['max_count']}"
    )
    print("-" * 70)

    for option_index, option in enumerate(request["options"]):
        useful_fields = {
            key: value for key, value in option.items() if value is not None
        }
        print(
            f"[{option_index}] "
            f"{json.dumps(useful_fields, ensure_ascii=False)}"
        )


def parse_human_input(raw_input: str) -> list[int]:
    normalized = raw_input.replace(",", " ").strip()

    if not normalized:
        return []

    return [int(token) for token in normalized.split()]


def validate_selection(
    selected_indices: list[int],
    request: Request,
) -> str | None:
    min_count = request["min_count"]
    max_count = request["max_count"]
    option_count = len(request["options"])
    selected_count = len(selected_indices)

    if not min_count <= selected_count <= max_count:
        return (
            f"You must select {min_count} to {max_count} options, "
            f"but selected {selected_count}."
        )

    if len(set(selected_indices)) != selected_count:
        return "The same option cannot be selected more than once."

    invalid_indices = [
        index
        for index in selected_indices
        if not isinstance(index, int) or not 0 <= index < option_count
    ]

    if invalid_indices:
        return (
            f"Invalid option indices: {invalid_indices}. "
            f"Valid range: 0 to {option_count - 1}."
        )

    return None


def get_human_decision(
    obs_dict: Observation,
    request: Request,
) -> list[int]:
    min_count = request["min_count"]
    max_count = request["max_count"]
    option_count = len(request["options"])

    if max_count == 0:
        print("No selection required. Automatically submitting [].")
        return []

    if option_count == 1 and min_count == 1 and max_count == 1:
        print("Only one legal option. Automatically selecting [0].")
        return [0]

    while True:
        if min_count == max_count:
            print(f"Enter exactly {min_count} option index/indices.")
        else:
            print(
                f"Enter between {min_count} and {max_count} option indices."
            )

        print("For multiple options, use spaces or commas.")

        if min_count == 0:
            print("Press Enter without typing anything to select none.")

        try:
            selected_indices = parse_human_input(input("Your selection: "))
        except ValueError:
            print("Invalid input: enter integers only.")
            continue

        error = validate_selection(selected_indices, request)

        if error is None:
            return selected_indices

        print(f"Invalid selection: {error}")


def make_human_agent(
    deck: list[int],
) -> Callable[[Observation], list[int]]:
    validate_deck(deck)
    agent_deck = deck.copy()
    decision_number = 0

    def agent(obs_dict: Observation) -> list[int]:
        nonlocal decision_number

        if obs_dict.get("select") is None:
            print("\nCABT requested the deck.")
            print(f"Submitting {len(agent_deck)} card IDs.")

            return agent_deck.copy()

        request = collect_options(obs_dict)
        decision_number += 1
        display_options(request, decision_number)
        selected_indices = get_human_decision(obs_dict, request)
        error = validate_selection(selected_indices, request)

        if error is not None:
            raise ValueError(f"Decision submission failed: {error}")

        print(f"submitted action: {selected_indices}")

        return selected_indices

    return agent
