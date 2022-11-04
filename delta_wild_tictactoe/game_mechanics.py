import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

import pygame

HERE = Path(__file__).parent.resolve()


####################### USEFUL FUNCTIONS #############################


def reward_function(board: List[str]) -> int:
    """This function returns the reward that wo be given to the player who played the most recent
    move on 'board'.

    Returns either 0 or 1
    """
    return int(is_winner(board))


def choose_move_randomly(board: List[str]) -> Tuple[int, str]:
    """This function takes a random move.

    It is an excellent first opponent to set as opponent_choose_move
    """
    position: int = random.choice([count for count, item in enumerate(board) if item == Cell.EMPTY])
    counter: str = random.choice([Cell.O, Cell.X])
    return position, counter


def place_counter(board: List[str], position: int, counter: str) -> List[str]:
    """Place counter into board at position.

    Returns a copy of the board so does not mutate the original in place.
    """
    board = board.copy()
    check_action_valid((position, counter), board)
    board[position] = counter

    return board


def play_wild_ttt_game(
    your_choose_move: Callable[[List[str]], Tuple[int, str]],
    opponent_choose_move: Callable[[List[str]], Tuple[int, str]],
    game_speed_multiplier: float = 1.0,
    verbose: bool = False,
    render: bool = False,
):
    """Play a game where moves are chosen by `your_choose_move()` and `opponent_choose_move()`. Who
    goes first is chosen at random.

    Args:
        your_choose_move: function that chooses move (takes board as input)
        opponent_choose_move: function that picks your opponent's next move
        game_speed_multiplier: multiplies the speed of the game. High == fast
        verbose: whether to print board states to console. For debugging

    Returns: total_return, which is the sum of return from the game
    """
    game = WildTictactoeEnv(
        opponent_choose_move,
        game_speed_multiplier=game_speed_multiplier,
        verbose=verbose,
        render=render,
    )

    state, reward, done, info = game.reset()
    while not done:

        action = your_choose_move(state)
        state, reward, done, info = game.step(action)


class Cell:
    """You will need to interact with this!

    This class represents the state of a single square of the tic-tac-toe  board. An X counter is
    represented by Cell.X An O counter is  represented by Cell.O A blank square represented by
    Cell.EMPTY
    """

    EMPTY = " "
    X = "X"
    O = "O"


############################ LESS USEFUL ##################################


class Player:
    """This class defines which players turn it is.

    If player: It is the turn of the player passing action directly to step.  If opponent: It is the
    turn of the opponent_choose_move function passed  to WildTictactoeEnv's __init__()
    """

    player = "player"
    opponent = "opponent"


def is_board_full(board: List[str]) -> bool:
    """Check if the board is full by checking for empty cells after flattening board."""
    return all(c != Cell.EMPTY for c in board)


def _check_winning_set(iterable: Iterable[str]) -> bool:
    unique_pieces = set(iterable)
    return Cell.EMPTY not in unique_pieces and len(unique_pieces) == 1


def is_winner(board: List[str]) -> bool:
    # 3x3 list of lists representing board
    nested_board = [board[i : i + 3] for i in range(0, len(board), 3)]

    # Check rows
    for row in nested_board:
        if _check_winning_set(row):
            return True

    # Check columns
    for column in [*zip(*nested_board)]:
        if _check_winning_set(column):
            return True

    # Check major diagonal
    size = len(nested_board)
    major_diagonal = [nested_board[i][i] for i in range(size)]
    if _check_winning_set(major_diagonal):
        return True

    # Check minor diagonal
    minor_diagonal = [nested_board[i][size - i - 1] for i in range(size)]
    return bool(_check_winning_set(minor_diagonal))


def get_empty_board() -> List[str]:
    return [Cell.EMPTY] * 9


def check_action_valid(action: Tuple[int, str], board: List[str]):

    assert isinstance(action, tuple), "Action must be a tuple of (position, counter)"
    assert isinstance(action[0], int), "Action[0] must be an integer"
    assert isinstance(action[1], str), "Action[1] must be a string"

    position, counter = action

    assert position in range(9), "Position must be between 0 and 8"
    assert (
        board[position] == Cell.EMPTY
    ), "You moved onto a square that already has a counter on it!"

    assert counter in {Cell.X, Cell.O}, "Counter must be either X or O"


class WildTictactoeEnv:
    def __init__(
        self,
        opponent_choose_move: Callable[[List], Tuple[int, str]] = choose_move_randomly,
        game_speed_multiplier: float = 1,
        verbose: bool = False,
        render: bool = False,
    ):
        self.opponent_choose_move = opponent_choose_move
        self.done: bool = False
        self.board = get_empty_board()
        self.verbose = verbose
        self.render = render
        self.game_speed_multiplier = game_speed_multiplier
        if self.render:
            self.screen = init_pygame()

    def __repr__(self) -> str:
        return str(np.array([x for xs in self.board for x in xs]).reshape((3, 3))) + "\n"

    def switch_player(self) -> None:
        self.player_move: str = (
            Player.player if self.player_move == Player.opponent else Player.opponent
        )

    def step(self, action: Tuple[int, str]) -> Tuple[List[str], int, bool, Dict]:
        """Called by user - takes 2 turns, yours and your opponent's"""

        reward = self._step(action)

        if not self.done:
            opponent_action = self.opponent_choose_move(self.board)
            opponent_reward = self._step(opponent_action)
            # Negative sign is because the opponent's victory is your loss
            reward -= opponent_reward

        if self.verbose:
            if reward == 1:
                print("You win!")
            elif reward == -1:
                print("Oh no, your opponent won!")
            elif self.done:
                print("Game Drawn!")

        return self.board, reward, self.done, {}

    def _step(self, action: Tuple[int, str]) -> int:

        assert not self.done, "Game is done. Call reset() before taking further steps."
        check_action_valid(action, self.board)

        position, counter = action

        self.board = place_counter(self.board, position, counter)
        if self.verbose:
            print(f"{self.player_move} makes a move!")
            print(self)

        self.counter_players[position] = self.player_move
        if self.render:
            self.render_game()

        winner = is_winner(self.board)
        board_full = is_board_full(self.board)
        reward = 1 if winner else 0
        self.done = winner or board_full

        self.switch_player()

        return reward

    def reset(self) -> Tuple[List[str], int, bool, Dict]:
        self.board = get_empty_board()

        self.done = False

        self.player_move = random.choice([Player.player, Player.opponent])
        self.went_first = self.player_move

        if self.verbose:
            print("Game starts!")
            print(self)

        self.counter_players: Dict[int, str] = {}

        if self.render:
            self.render_game()

        if self.player_move == Player.opponent:
            opponent_action = self.opponent_choose_move(self.board)
            reward = -self._step(opponent_action)
            if self.render:
                self.render_game()
        else:
            reward = 0

        return self.board, reward, self.done, {}

    def render_game(self):
        render(self.screen, self.board, self.counter_players, self.player_move)
        time.sleep(1 / self.game_speed_multiplier)


######## Do not worry about anything below here ###################


WIDTH = 600
HEIGHT = 600
LINE_WIDTH = 15
WIN_LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = 200
CIRCLE_RADIUS = 60
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = 55

RED = (255, 0, 0)
BG_COLOR = (26, 28, 31)
LINE_COLOR = (255, 255, 255)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)


PLAYER_COLORS = {"player": "blue", "opponent": "red"}


def draw_pieces(screen, board: List[str], counter_players: Dict) -> None:
    # Draw circles and crosses based on board state

    for position, counter in enumerate(board):
        col = position % 3
        row = position // 3
        if counter == Cell.EMPTY:
            continue
        color = PLAYER_COLORS[counter_players[position]]
        if counter == Cell.O:
            pygame.draw.circle(
                screen,
                color,
                (
                    int(col * SQUARE_SIZE + SQUARE_SIZE // 2),
                    int(row * SQUARE_SIZE + SQUARE_SIZE // 2),
                ),
                CIRCLE_RADIUS,
                CIRCLE_WIDTH,
            )

        elif counter == Cell.X:
            pygame.draw.line(
                screen,
                color,
                (
                    col * SQUARE_SIZE + SPACE,
                    row * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                ),
                (
                    col * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                    row * SQUARE_SIZE + SPACE,
                ),
                CROSS_WIDTH,
            )
            pygame.draw.line(
                screen,
                color,
                (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE),
                (
                    col * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                    row * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                ),
                CROSS_WIDTH,
            )


def check_and_draw_win(board: List, counter: str, screen: pygame.Surface, player_move: str) -> bool:

    for col in range(BOARD_COLS):
        if all(board[idx] == counter for idx in [0 + col, 3 + col, 6 + col]):
            draw_vertical_winning_line(screen, col, player_move)
            return True

    for row in range(BOARD_ROWS):
        if all(board[idx] == counter for idx in [0 + row * 3, 1 + row * 3, 2 + row * 3]):
            draw_horizontal_winning_line(screen, row, player_move)
            return True

    if all(board[idx] == counter for idx in [0, 4, 8]):
        draw_desc_diagonal(screen, player_move)
        return True

    if all(board[idx] == counter for idx in [2, 4, 6]):
        draw_asc_diagonal(screen, player_move)
        return True

    return False


def draw_vertical_winning_line(screen, col, player_move):
    posX = col * SQUARE_SIZE + SQUARE_SIZE // 2
    team_color = PLAYER_COLORS[player_move]

    pygame.draw.line(
        screen,
        team_color,
        (posX, 15),
        (posX, HEIGHT - 15),
        LINE_WIDTH,
    )


def draw_horizontal_winning_line(screen, row, player_move):
    posY = row * SQUARE_SIZE + SQUARE_SIZE // 2

    team_color = PLAYER_COLORS[player_move]
    pygame.draw.line(
        screen,
        team_color,
        (15, posY),
        (WIDTH - 15, posY),
        WIN_LINE_WIDTH,
    )


def draw_asc_diagonal(screen, player_move):
    team_color = PLAYER_COLORS[player_move]
    pygame.draw.line(
        screen,
        team_color,
        (15, HEIGHT - 15),
        (WIDTH - 15, 15),
        WIN_LINE_WIDTH,
    )


def draw_desc_diagonal(screen, player_move):
    team_color = PLAYER_COLORS[player_move]
    pygame.draw.line(
        screen,
        team_color,
        (15, 15),
        (WIDTH - 15, HEIGHT - 15),
        WIN_LINE_WIDTH,
    )


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TIC TAC TOE")
    return screen


def render(screen, board: List, counter_players: Dict[int, str], player_move: str):

    screen.fill(BG_COLOR)

    # DRAW LINES
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (0, 2 * SQUARE_SIZE),
        (WIDTH, 2 * SQUARE_SIZE),
        LINE_WIDTH,
    )
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (2 * SQUARE_SIZE, 0),
        (2 * SQUARE_SIZE, HEIGHT),
        LINE_WIDTH,
    )

    draw_pieces(screen, board, counter_players)

    for counter in [Cell.X, Cell.O]:
        check_and_draw_win(board, counter, screen=screen, player_move=player_move)

    pygame.display.update()


def pos_to_coord(pos: Tuple[int, int]):
    n_rows = 3
    # Assume square board
    square_size = WIDTH / n_rows

    col = math.floor(pos[0] / square_size)
    row = math.floor(pos[1] / square_size)
    return row, col


def coord_to_action(coord: Tuple[int, int]):
    return coord[0] * 3 + coord[1]


LEFT = 1
RIGHT = 3


def human_player(state) -> Tuple[int, str]:
    print(
        "Your move, click to place a tile!\nLeft click to place an `O`.\nRight click to place an `X`."
    )

    while True:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                coord = pos_to_coord(pos)
                square = coord_to_action(coord)

                if event.button == RIGHT:
                    return (square, Cell.X)
                if event.button == LEFT:
                    return (square, Cell.O)


def save_dictionary(my_dict: Dict, team_name: str) -> None:
    assert isinstance(
        my_dict, dict
    ), f"train() function should output a dict, but got: {type(my_dict)}"
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"

    n_retries = 5
    dict_path = os.path.join(HERE, f"dict_{team_name}.pkl")
    for attempt in range(n_retries):
        try:
            with open(dict_path, "wb") as f:
                pickle.dump(my_dict, f)
            load_dictionary(team_name)
            return
        except Exception as e:
            if attempt == n_retries - 1:
                raise


def load_dictionary(team_name: str, umbrella: Path = HERE) -> Dict:
    dict_path = os.path.join(umbrella, f"dict_{team_name}.pkl")
    with open(dict_path, "rb") as f:
        return pickle.load(f)
