from typing import List, Tuple

from delta_wild_tictactoe.game_mechanics import Cell, Player, WildTictactoeEnv


def choose_move_X(board: List[str]) -> Tuple[int, str]:
    """Puts an X on the first blank tile on the board."""
    for idx in range(len(board)):
        if board[idx] == " ":  # Slightly upside down logic to keep mypy happy
            break
    return idx, Cell.X


def choose_move_O(board: List[str]) -> Tuple[int, str]:
    """Puts an O on the first blank tile on the board."""
    for idx in range(len(board)):
        if board[idx] == " ":  # Slightly upside down logic to keep mypy happy
            break
    return idx, Cell.O


def _test_wild_tictactoe_env() -> None:
    """Tests WildTictactoeEnv by playing through a game with deterministic players Possible TODO:

    more unit-testy structure where individual methods tested separately.
    """
    your_choose_move = choose_move_X
    opponent_choose_move = choose_move_O

    game = WildTictactoeEnv(opponent_choose_move)
    state, reward, done, info = game.reset()
    assert reward == 0.0
    assert done == False
    assert len(info) == 0

    # Reset takes first opponent move if they go first
    if game.went_first == Player.opponent:
        assert state[0] == Cell.O

    # First step call, takes two moves
    action = your_choose_move(state)
    state, reward, done, info = game.step(action)
    assert reward == 0.0
    assert done == False

    if game.went_first == Player.player:
        assert state[0] == Cell.X
        assert state[1] == Cell.O
    else:
        assert state[0] == Cell.O
        assert state[1] == Cell.X
        assert state[2] == Cell.O

    # Second step call, takes two moves
    action = your_choose_move(state)
    state, reward, done, info = game.step(action)
    assert reward == 0.0
    assert done == False
    assert len(info) == 0

    if game.went_first == "player":
        assert state[0] == Cell.X
        assert state[1] == Cell.O
        assert state[2] == Cell.X
        assert state[3] == Cell.O
    else:
        assert state[0] == Cell.O
        assert state[1] == Cell.X
        assert state[2] == Cell.O
        assert state[3] == Cell.X
        assert state[4] == Cell.O

    assert state[5] == Cell.EMPTY

    # Third step call, takes two moves opponent wins if first
    action = your_choose_move(state)
    state, reward, done, info = game.step(action)
    assert len(info) == 0

    if game.went_first == "player":
        assert state[0] == Cell.X
        assert state[1] == Cell.O
        assert state[2] == Cell.X
        assert state[3] == Cell.O
        assert state[4] == Cell.X
        assert state[5] == Cell.O
        assert done == False
        assert reward == 0.0
    else:
        assert state[0] == Cell.O
        assert state[1] == Cell.X
        assert state[2] == Cell.O
        assert state[3] == Cell.X
        assert state[4] == Cell.O
        assert state[5] == Cell.X
        assert state[6] == Cell.O
        assert state[7] == Cell.EMPTY

        assert done == True
        assert reward == -1.0

    # Fourth step call, only if player first
    if game.went_first == "player":
        action = your_choose_move(state)
        state, reward, done, info = game.step(action)
        assert len(info) == 0
        assert state[0] == Cell.X
        assert state[1] == Cell.O
        assert state[2] == Cell.X
        assert state[3] == Cell.O
        assert state[4] == Cell.X
        assert state[5] == Cell.O
        assert state[6] == Cell.X
        assert state[7] == Cell.EMPTY
        assert done
        assert reward == 1.0
    else:
        assert done


def test_wild_tictactoe_env() -> None:
    """Run the above test 10 times as randomness in the first player.

    A better way of doing this might be to set the first player then test, rather than relying on
    the randomness
    """
    for _ in range(10):
        _test_wild_tictactoe_env()
