from collections import namedtuple
import torch


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

Bin = namedtuple('Bin', ('width', 'height', 'weight'))

Action = namedtuple('Action', ('bin_index', 'priority', 'rotate'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GUI:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)

    # WINDOW_POS_X = 100
    # WINDOW_POS_Y = 100

class ENV:

    RENDER = True
    TICK_INTERVAL = 100  # the smaller it is, the slower the game plays 100

    # Environment Parameter
    FONT_SIZE = 15
    ROW_COUNT = 10
    COL_COUNT = 10
    CELL_SIZE = 50
    # CAPTION_NAME = '2D Bin Packing simulator by Uk Jo'

    # Bins information
    BIN_N_STATE = 3  # pos, weight on 2d
    BIN_MAX_COUNT = 30


    BIN_MIN_X_SIZE = 1
    BIN_MAX_X_SIZE = 3
    BIN_MIN_Y_SIZE = 1
    BIN_MAX_Y_SIZE = 3
    BIN_MIN_W_SIZE = 1
    BIN_MAX_W_SIZE = 1

    # Agent Side
    AGENT_STARTING_POS = [0, 0]
    ACTION_SIZE = 3   # x, y, rotate
    N_EPISODES = 1000000  # 最大回合
    EPISODE_MAX_STEP = 400  # 每回合最大步数
    # Constraint
    LOAD_WIDTH_THRESHOLD = 0.8  # Ratio

    ACTION_SPACE = (BIN_MAX_COUNT, 2, 2)
    VERBOSE = 1


class REWARD:
    INVALID_ACTION_REWARD = 0
    GOAL_REWARD = 1.0
    MODE = 1


class AGENT:
    DISCOUNT_FACTOR_REWARD = 0.9
    LEARNING_RATE = 0.01
    EPSILON = 0.9
    BATCH_SIZE = 32
    TARGET_UPDATE_INTERVAL = 1000
    GAMMA = 0.9
    REPLAY_MEMORY_SIZE = 10000
    EMBEDDING_DIM = 10
