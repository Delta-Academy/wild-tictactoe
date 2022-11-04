import random
from typing import List

import pytest
from delta_wild_tictactoe.game_mechanics import (
    Cell,
    check_action_valid,
    choose_move_randomly,
    get_empty_board,
    place_counter,
    reward_function,
)


def check_board_correct(board: List[str]):
    assert len(board) == 9
    assert isinstance(board, list)
    assert all(isinstance(x, str) for x in board)
    assert all(x in {Cell.X, Cell.O, Cell.EMPTY} for x in board)


def test_get_empty_board():
    board = get_empty_board()
    check_board_correct(board)
    assert all(x == Cell.EMPTY for x in board)


def test_reward_function():
    board = get_empty_board()
    assert reward_function(board) == 0
    board[0] = Cell.X
    check_board_correct(board)
    assert reward_function(board) == 0
    board[1] = Cell.X
    check_board_correct(board)
    assert reward_function(board) == 0
    board[2] = Cell.X
    check_board_correct(board)
    assert reward_function(board) == 1

    board = get_empty_board()
    assert reward_function(board) == 0
    board[0] = Cell.O
    check_board_correct(board)
    assert reward_function(board) == 0
    board[1] = Cell.O
    check_board_correct(board)
    assert reward_function(board) == 0
    board[2] = Cell.X
    check_board_correct(board)
    assert reward_function(board) == 0
    board[3] = Cell.O
    check_board_correct(board)
    assert reward_function(board) == 0
    board[4] = Cell.X
    check_board_correct(board)
    assert reward_function(board) == 0
    board[6] = Cell.O
    check_board_correct(board)
    assert reward_function(board) == 1
    board[6] = Cell.EMPTY
    check_board_correct(board)
    assert reward_function(board) == 0
    board[6] = Cell.X
    check_board_correct(board)
    assert reward_function(board) == 1


def test_place_counter_valid_actions():
    board = get_empty_board()
    new_board = place_counter(board, 0, Cell.X)
    assert new_board[0] == Cell.X
    assert all(x == Cell.EMPTY for x in new_board[1:])
    assert all(x == Cell.EMPTY for x in board)

    new_board = place_counter(new_board, 1, Cell.O)
    assert all(x == Cell.EMPTY for x in board)
    assert all(x == Cell.EMPTY for x in new_board[2:])
    assert new_board[0] == Cell.X
    assert new_board[1] == Cell.O

    new_board = place_counter(new_board, 8, Cell.X)
    assert all(x == Cell.EMPTY for x in board)
    assert new_board[0] == Cell.X
    assert new_board[1] == Cell.O
    assert new_board[8] == Cell.X


def test_place_counter_invalid_actions():
    board = get_empty_board()
    new_board = place_counter(board, 0, Cell.X)
    with pytest.raises(AssertionError):
        # Place counter on top of existing counter
        place_counter(new_board, 0, Cell.X)

    with pytest.raises(AssertionError):
        # Place empty counter
        place_counter(new_board, 0, Cell.EMPTY)

    with pytest.raises(AssertionError):
        # position too big
        place_counter(new_board, 10, Cell.X)

    with pytest.raises(AssertionError):
        # position too small
        place_counter(new_board, -1, Cell.X)

    with pytest.raises(AssertionError):
        # action not an int
        place_counter(new_board, 1.5, Cell.X)  # type: ignore


def get_random_board():
    board = get_empty_board()
    for position in range(9):
        counter = random.choice([Cell.X, Cell.O, Cell.EMPTY])
        board[position] = counter
    return board


def test_choose_move_randomly():
    for _ in range(100):
        board = get_random_board()
        if Cell.EMPTY not in board:
            continue
        action = choose_move_randomly(board)
        check_action_valid(action=action, board=board)
