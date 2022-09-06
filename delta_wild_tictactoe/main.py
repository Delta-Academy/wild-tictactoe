import random
from typing import Dict, List, Tuple

from check_submission import check_submission
from game_mechanics import (
    Cell,
    WildTictactoeEnv,
    choose_move_randomly,
    human_player,
    load_dictionary,
    play_wild_ttt_game,
    render,
    save_dictionary,
)

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> Dict:
    """Write this function to train your algorithm.

    Returns:
         Value function dictionary used by your agent. You can
         structure this how you like, however your choose_move must
         be able to use it.
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(board: List[str], value_function: Dict) -> Tuple[int, str]:
    """
    TODO: WRITE THIS FUNCTION

    This is what will be called during competitive play.
    It takes the current state of the board as input.
    It returns a single move to play.

    Args:
        board: list representing the board.
                (see README Technical Details for more info)

        value_function: The dictionary output by train().

    Returns:
        position (int): The position to place your piece
                        (an integer 0 -> 8), where 0 is
                        top left and 8 is bottom right.
        counter (str): The counter to place. "X" or "O".

    It's important that you think about exactly what this
     function does when you submit, as it will be called
     in order to take your turn!
    """

    # We provide an example here that chooses a random position on the board and
    # and places a random counter there.
    # position = random.choice([count for count, item in enumerate(board) if item == Cell.EMPTY])
    # counter = random.choice([Cell.O, Cell.X])
    # return position, counter
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    my_value_fn = train()
    save_dictionary(my_value_fn, TEAM_NAME)

    # Make sure I do not error, or your submission will
    # not work in the competition
    check_submission(TEAM_NAME)

    my_value_fn = load_dictionary(TEAM_NAME)

    def choose_move_no_value_fn(board: List[str]) -> Tuple[int, str]:
        """The arguments in play_wild_ttt_game() require functions that only take the state as
        input.

        This converts choose_move() to that format.
        """
        return choose_move(board, my_value_fn)

    # Code below plays a single game of Wild Tic-Tac-Toe vs a random
    # opponent, think about how you might want to adapt this to
    # test the performance of your algorithm.

    # Play against your bot!!
    play_wild_ttt_game(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_value_fn,
        game_speed_multiplier=1,
        verbose=True,
        render=True,
    )
