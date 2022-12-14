from collections import defaultdict
from pathlib import Path
from typing import Dict

import delta_utils.check_submission as checker
from game_mechanics import get_empty_board, load_dictionary


def pkl_checker_value_dict(pkl_file: Dict) -> None:
    """Checks a dictionary acting as a value lookup table."""
    if isinstance(pkl_file, defaultdict):
        assert not callable(
            pkl_file.default_factory
        ), "Please don't use functions within default dictionaries in your pickle file!"

    assert len(pkl_file) > 0, "Your dictionary is empty!"

    for v in pkl_file.values():
        assert isinstance(
            v, (float, int)
        ), "Your value function dictionary values should be a number!"


def check_submission(team_name: str) -> None:
    example_state = get_empty_board()
    expected_choose_move_return_type = tuple
    expected_pkl_output_type = Dict

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=load_dictionary(team_name),
        pkl_checker_function=pkl_checker_value_dict,
        current_folder=Path(__file__).parent.resolve(),
    )
