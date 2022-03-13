from gym.core import Env as Env
from env_config import *
import numpy as np
import random
import os
import pandas as pd
import logging
import pickle

logger = logging.getLogger('uk')
logger.setLevel(logging.ERROR)

if ENV.RENDER:
    import pygame


class PalleteWorld(Env):
    """
    State : (Bin List, Current Snapshot of 2D Pallet (exist, weight) )
    """

    def __init__(self, mode='agent', n_random_fixed=None, datasets=None, env_id=None):

        self.env_id = env_id
        self.n_random_fixed = n_random_fixed
        self.total_items = []
        self.placed_items = []
        self.total_pixel_row_size = ENV.CELL_SIZE * ENV.ROW_COUNT  # 创建屏幕大小
        self.total_pixel_col_size = ENV.CELL_SIZE * ENV.COL_COUNT
        self.bins_list = []
        self.p = []
        self.current_bin_placed = 0  # 当前放置的bin个数
        self.previous_actions = []  # 放过的bin索引
        self.placed_items = []  # 放置的bin
        self.is_bin_placed = False
        self.n_step = 0
        self.percentage = 0
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.init()
        self.screen = pygame.display.set_mode([self.total_pixel_row_size, self.total_pixel_col_size])
        self.screen.fill(GUI.WHITE)
        self.font = pygame.font.SysFont('consolas', ENV.FONT_SIZE, 1)

        i = 0
        self.colors = []  # BIN_MAX_COUNT 个颜色
        while i < ENV.BIN_MAX_COUNT:
            c = pygame.color.THECOLORS.popitem()[1][0:3]
            if c == GUI.WHITE:
                pass
            else:
                self.colors.append(c)
                i = i + 1
        # 创建bins
        for i in range(self.n_random_fixed):
            bins_list = []
            for i in range(ENV.BIN_MAX_COUNT):
                x = random.randint(ENV.BIN_MIN_X_SIZE, ENV.BIN_MAX_X_SIZE)
                y = random.randint(ENV.BIN_MIN_Y_SIZE, ENV.BIN_MAX_Y_SIZE)
                w = random.randint(ENV.BIN_MIN_W_SIZE, ENV.BIN_MAX_W_SIZE)
                bins_list.append((x, y, w))
            self.total_items.append(bins_list)
        self.reset()

    def reset(self):
        logging.debug('env reset')

        self.p = np.zeros((ENV.ROW_COUNT, ENV.COL_COUNT, 3))
        self.p[:, :, 2] = -1
        self.bins_list = self.total_items[0].copy()
        self.current_bin_placed = 0  # 当前放置的bin个数
        self.previous_actions = []  # 放过的bin索引
        self.placed_items = []  # 放置的bin
        self.is_bin_placed = False
        self.n_step = 0
        # 返回的初始observation 包裹生成的bins_list 与 p = np.zeros((ENV.ROW_COUNT, ENV.COL_COUNT, 3))
        return self.bins_list, self.p

    def is_valid(self, c, r, b: Bin):
        is_exist = False
        is_fall = False
        can_load = True
        is_spacious = True
        is_exist_above = False
        verbose = ENV.VERBOSE

        # todo : for test, it examines every possible cases even if it finds out is not impossible for now.
        if r + b.height > ENV.ROW_COUNT:  # 是否超出方框
            is_spacious = False
            if verbose != 0:
                logger.debug('Not valid. It\'s not spacious over x axis.')
            return False

        if c + b.width > ENV.COL_COUNT:
            is_spacious = False
            if verbose != 0:
                logger.debug('Not valid. It\'s not spacious over y axis.')
            return False

        # todo : to make it real, it adds constraints that any bins can not load if there is any bins above of it.
        if r != ENV.ROW_COUNT - 1:
            if not (0. == self.p[0:ENV.ROW_COUNT - r, c:c + b.width, 0]).all():  #
                if verbose != 0:
                    logger.debug('this is not valid because there are other bins above of it.')
                is_exist_above = True
                return False

        if not (0. == self.p[ENV.ROW_COUNT - r - b.height:ENV.ROW_COUNT - r, c:c + b.width, 0]).all():
            if verbose != 0:
                logger.debug('Not valid. The area is already possessed by other bins.')
            is_exist = True
            return False

        if r >= 1:
            if (b.width - sum(
                    (self.p[ENV.ROW_COUNT - r, c:c + b.width, 0]) == 0.)) / b.width >= ENV.LOAD_WIDTH_THRESHOLD:
                # todo : check this constraint working
                is_fail = False  # fall
            else:
                if verbose != 0:
                    logger.debug('Not val11id. The bin could be fallen lack of support of bottom bins')
                is_fall = True
                return False

        # can_load check
        # if b.weight > self.p[r:r+b.height, c - 1, 1].any():
        #     if verbose != 0:
        #         logger.debug('Not valid. The bins below can not stand the weight of this bin.')
        #     can_load = False
        #     return False

        if is_spacious and not is_exist and not is_fall and can_load and not is_exist_above:
            return True
        else:
            return False

    def step(self, action: Action):
        """
        :param action: bin index, priority x or priority y, rotate
        :return: observation, reward, done, info
        """
        self.is_bin_placed = False
        self.n_step += 1

        if action.bin_index not in self.previous_actions:

            b = Bin(*(self.bins_list[action.bin_index]))

            if action.rotate == 1:
                b = Bin(*(b.height, b.width, b.weight))

            # https://stackoverflow.com/questions/2597104/break-the-nested-double-loop-in-python
            class BinPlaced(Exception):
                pass

            try:
                if action.priority == 0:
                    for y in range(0, ENV.ROW_COUNT):
                        for x in range(0, ENV.COL_COUNT):
                            if self.is_valid(x, y, b):
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 0] = 1
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 1] = b.weight
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width,
                                2] = action.bin_index
                                raise BinPlaced
                elif action.priority == 1:
                    for x in range(0, ENV.COL_COUNT):
                        for y in range(0, ENV.ROW_COUNT):
                            if self.is_valid(x, y, b):
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 0] = 1
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 1] = b.weight
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width,
                                2] = action.bin_index
                                raise BinPlaced
                if ENV.VERBOSE != 0:
                    logging.debug('This bin can not be placed.')
            except BinPlaced:
                self.is_bin_placed = True
                self.current_bin_placed += 1
                self.placed_items.append(action.bin_index)

        # TODO : We can add this only when this box is replaced..
        self.previous_actions.append(action.bin_index)

        # mask bins_list
        for i in self.previous_actions:
            self.bins_list[i] = (0, 0, 0)

        board = np.asarray(self.p[:, :, 0])
        area = board.sum()
        total_area = ENV.ROW_COUNT * ENV.COL_COUNT

        self.percentage = area / total_area

        # next_state, reward, done, info          # self.p = np.zeros((ENV.ROW_COUNT, ENV.COL_COUNT, 3))
        return (self.bins_list, self.p), self.get_reward(), self.is_done(), {'placed_items': self.placed_items,
                                                                             'percentage': self.percentage}

    def is_done(self):

        # print('percentage : {}'.format(percentage))

        if self.current_bin_placed == ENV.BIN_MAX_COUNT or self.n_step == ENV.EPISODE_MAX_STEP or len(
                self.previous_actions) == ENV.BIN_MAX_COUNT and self.percentage > 0.9:
            print('count', self.current_bin_placed, 'step', self.n_step, 'pre', len(
                self.previous_actions), 'percentage', self.percentage)
            return True
        else:
            return False

    def get_reward(self):
        """
        :return:
        """
        # todo : do more reward engineering for every step
        if REWARD.MODE == 0:
            if self.is_done():
                return self.current_bin_placed / ENV.BIN_MAX_COUNT
            elif not self.is_bin_placed:
                return REWARD.INVALID_ACTION_REWARD
            else:
                return 0
        elif REWARD.MODE == 1:  # sum of placed bins' space
            if self.is_done():
                board = np.asarray(self.p[:, :, 0])
                return board.sum()
            elif not self.is_bin_placed:
                return REWARD.INVALID_ACTION_REWARD
            else:
                return 0
        elif REWARD.MODE == 2:  # number of placed bins
            if self.is_done():
                return len(len((self.p[:, :, 0]) != 0.))
            elif not self.is_bin_placed:
                return REWARD.INVALID_ACTION_REWARD
            else:
                return 0

    def render(self, mode='agent'):
        self.screen.fill(GUI.WHITE)  # 没用
        for y in range(0, ENV.ROW_COUNT):
            for x in range(0, ENV.COL_COUNT):
                index = int(self.p[y, x, 2])
                if index == -1:
                    pass
                else:
                    pygame.draw.rect(
                        self.screen,
                        self.colors[index],
                        [x * ENV.CELL_SIZE, y * ENV.CELL_SIZE, ENV.CELL_SIZE, ENV.CELL_SIZE],
                    )

        pygame.display.flip()

    def close(self):
        if ENV.RENDER:
            pygame.quit()

    def get_action(self, state):
        pass


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        self.pieces = {'1': 'X', '0': '-', '-1': 'O'}
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    def _allowedActions(self):
        allowed = []
        for i in range(len(self.board)):
            if i >= len(self.board) - 7:
                if self.board[i] == 0:
                    allowed.append(i)
            else:
                if self.board[i] == 0 and self.board[i + 7] != 0:
                    allowed.append(i)

        return allowed

    def _binary(self):
        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board == self.playerTurn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -self.playerTurn] = 1

        position = np.append(currentplayer_position, other_position)

        return (position)

    def _convertStateToId(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id = ''.join(map(str, position))

        return id

    def _checkForEndGame(self):
        if np.count_nonzero(self.board) == 42:
            return 1

        for x, y, z, a in self.winners:
            if (self.board[x] + self.board[y] + self.board[z] + self.board[a] == 4 * -self.playerTurn):
                return 1
        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        for x, y, z, a in self.winners:
            if (self.board[x] + self.board[y] + self.board[z] + self.board[a] == 4 * -self.playerTurn):
                return (-1, -1, 1)
        return (0, 0, 0)

    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])

    def takeAction(self, action):
        newBoard = np.array(self.board)
        newBoard[action] = self.playerTurn

        newState = GameState(newBoard, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def render(self, logger):
        for r in range(6):
            logger.info([self.pieces[str(x)] for x in self.board[7 * r: (7 * r + 7)]])
        logger.info('--------------')
