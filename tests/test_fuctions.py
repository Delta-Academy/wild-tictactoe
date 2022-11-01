import copy
import random
from typing import List

from delta_wild_tictactoe.game_mechanics import (
    Cell,
    check_action_valid,
    choose_move_randomly,
    get_empty_board,
    mark_square,
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


def test_mark_square_copy_board():
    board = get_empty_board()
    for _ in range(100):
        position = random.randint(0, 8)
        counter = random.choice([Cell.X, Cell.O, Cell.EMPTY])
        board_store = copy.deepcopy(board)
        new_board = mark_square(board, position, counter, copy_board=True)
        assert board == board_store
        assert new_board[position] == counter


def test_mark_square_do_not_copy():
    board = get_empty_board()
    for _ in range(100):
        position = random.randint(0, 8)
        counter = random.choice([Cell.X, Cell.O, Cell.EMPTY])
        new_board = mark_square(board, position, counter, copy_board=False)
        assert new_board[position] == counter


def get_random_board():
    board = get_empty_board()
    for position in range(9):
        counter = random.choice([Cell.X, Cell.O, Cell.EMPTY])
        board = mark_square(board, position, counter, copy_board=False)
    return board


def test_choose_move_randomly():
    for _ in range(100):
        board = get_random_board()
        if Cell.EMPTY not in board:
            continue
        action = choose_move_randomly(board)
        check_action_valid(action=action, board=board)
