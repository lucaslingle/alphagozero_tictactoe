# MCTS using UCT

import numpy as np
from collections import namedtuple
import tensorflow as tf
from random import shuffle
import time

GameStateTuple = namedtuple('GameStateTuple', ['board', 'turn_parity'])

class TurnParity:
    def __init__(self):
        self.sign = 1

    def flip(self):
        self.sign *= -1

    def __eq__(self, other):
        return self.sign == other

    def __repr__(self):
        return str(self.sign)

class TicTacToe:

    def __init__(self):
        self.state = GameStateTuple(
            board=np.zeros(dtype=np.int32, shape=(3,3)),
            turn_parity=TurnParity())

        self.episode_done = False

    def move_is_valid(self, move, player_id):
        if self.episode_done:
            return False

        if self.state.turn_parity != player_id:
            return False

        i = move[0]
        j = move[1]

        if self.state.board[i][j] != 0:
            return False

        return True

    def update_state(self, move, player_id):
        i = move[0]
        j = move[1]
        self.state.board[i][j] = player_id
        self.state.turn_parity.flip()

    def check_win(self):
        board = self.state.board

        columnar_victory = np.max(np.abs(np.sum(board, axis=0)) == 3)
        row_victory = np.max(np.abs(np.sum(board, axis=1)) == 3)
        forward_slash_victory = np.abs(board[2][0] + board[1][1] + board[0][2]) == 3
        back_slash_victory = np.abs(board[0][0] + board[1][1] + board[2][2]) == 3
        
        if columnar_victory or row_victory or forward_slash_victory or back_slash_victory:
            return True

        return False

    def check_full(self):
        board = self.state.board

        if np.sum(np.abs(board)) == 9:
            return True
        return False

    def step(self, move, player_id):
        '''
        Step the environment

        :param str move: tuple with row index and column index of a move
        :param str player_id: the ID of the player making the move, either -1 or 1
        :return: tuple with updated state, reward from perspective of player making the move, and indicator about whether game is over
        '''
        assert not self.episode_done
        assert self.move_is_valid(move, player_id)

        self.update_state(move, player_id)

        win = self.check_win()
        full = self.check_full()

        if win:
            reward = 1.0
            self.episode_done = True
        elif full and not win:
            reward = 0.0
            self.episode_done = True
        else:
            reward = 0.0

        return self.state, reward, self.episode_done

    def reset(self):
        '''
        Reset the environment
        '''
        self.state = GameStateTuple(
            board=np.zeros(dtype=np.int32, shape=(3,3)),
            turn_parity=TurnParity())

        self.episode_done = False

    def __repr__(self):
        if self.episode_done:
            line1 = str(-1 * self.state.turn_parity.sign) + ' has won.'
        else:
            line1 = str(self.state.turn_parity) + ' to move:'

        line2 = str(self.state.board)
        return line1 + '\n\n' + line2

    def clone(self):
        clone_env = TicTacToe()

        board = np.zeros(dtype=np.int32, shape=(3,3))
        for i in range(0,3):
            for j in range(0,3):
                board[i][j] = self.state.board[i][j]

        turn_parity = TurnParity()

        if self.state.turn_parity == -1:
            turn_parity.flip()

        state = GameStateTuple(
            board=board,
            turn_parity=turn_parity)

        clone_env.state = state
        return clone_env



game = TicTacToe()

print(game)
game.step(move=(1,1), player_id=1)
game.step(move=(1,2), player_id=-1)
game.step(move=(0,0), player_id=1)
game.step(move=(0,2), player_id=-1)
game.step(move=(2,2), player_id=1)
print(game)

game.reset()

print(game)
game.step(move=(1,1), player_id=1)
game.step(move=(1,2), player_id=-1)
game.step(move=(0,0), player_id=1)
game.step(move=(0,2), player_id=-1)
game.step(move=(2,0), player_id=1)
game.step(move=(2,2), player_id=-1)
print(game)



class Node:
    def __init__(self, s, parent_node_and_prev_action=None, is_root=False):
        # N, Q, P are vectors whose i-th coordinate corresponds to the i-th action that can be taken in state s.
        self.parent_node = None if parent_node_and_prev_action is None else parent_node_and_prev_action[0]
        self.prev_action = None if parent_node_and_prev_action is None else parent_node_and_prev_action[1]

        self.s = s.clone()
        self.Vs = None
        self.n = 1

        self.valid_actions_mask_vec = np.squeeze(np.reshape(self.get_valid_actions_mask(), [1, 9]), [0])
        self.Nas_vec = np.zeros(dtype=np.int32, shape=(9,))
        self.Was_vec = np.zeros(dtype=np.float32, shape=(9,))
        self.Qas_vec = np.zeros(dtype=np.float32, shape=(9,))
        self.Pas_vec = np.zeros(dtype=np.float32, shape=(9,))

        # hashtable of pointers to child nodes
        # the hashtable is keyed by integers, each representing the location of each possible move in a 1D representation of the board
        self.child_nodes = {i: None for i in range(0,9)}

        self.is_root = is_root

    def get_valid_actions_mask(self):
        if self.s.episode_done:
            return np.zeros(dtype=np.int32, shape=(3,3))

        bools_ = self.s.state.board == 0
        ints_ = np.array(bools_, dtype=np.int32)
        ints_ = np.reshape(ints_, [9])
        return ints_

    @property
    def is_leaf_node(self):
        return np.sum(self.Nas_vec) == 0 and not self.is_root

    @property
    def is_terminal_node(self):
        return self.s.episode_done

    @property
    def has_parent_node(self):
        return self.parent_node is not None

    def get_value_for_terminal_node(self):
        if self.is_terminal_node:
            # for terminal nodes, where game is actually over, use actual game outcome instead of neural network's value estimate

            env = self.s
            win = env.check_win()
            full = env.check_full()

             # for this implementation of tic tac toe, if terminal node involved a win, previous guy to move was the winner
            if win:
                v = -1.0
            elif full and not win:
                v = 0.0

            return v
        else:
            raise RuntimeError

    def predict_value_and_prior_vec_for_leaf_node(self, sess, f_theta):
        state = self.s.state
        state_repr = state.board * state.turn_parity.sign

        state_repr_batch = np.array([state_repr])
        valid_actions_mask_batch = np.array([self.valid_actions_mask_vec])

        p_vec, v = f_theta.predict(sess, state_repr_batch, valid_actions_mask_batch)

        return p_vec, v


def get_me_a_tree(state):
    node = Node(s=state, is_root=True)
    p_vec_leaf, v_leaf = node.predict_value_and_prior_vec_for_leaf_node(sess, f_theta)
    node.Pas_vec = p_vec_leaf
    node.Vs = v_leaf

    eps = 0.25

    #alpha = 0.03  # for go
    alpha = 1.21  # for tictactoe

    # per the deepmind alphazero paper, alpha is scaled in inverse proportion to avg num legal moves: 
    # so for go's alpha to be alpha = 0.03, we have 0.03 = c * 1/(362/2) for some c,
    # and thus c = 0.03 * (362/2), 
    # so that for tictactoe, alpha is approximately [0.03 * (362/2)] / (9/2) ~= 1.21

    alpha_vec = alpha * np.ones(shape=(node.Pas_vec.shape[1],))
    dirichlet = np.random.dirichlet(alpha_vec)
    node.Pas_vec = eps * dirichlet + (1. - eps) * node.Pas_vec

    return node


def run_me_a_simulation(tree, sess, f_theta):
    node = tree
    c_puct = np.sqrt(2.0)

    # search til we reach a leaf or a terminal node
    while not node.is_leaf_node and not node.is_terminal_node:
        Pas_vec = node.Pas_vec
        Qas_vec = node.Qas_vec
        Nas_vec = node.Nas_vec
        n = float(node.n)
        Uas_vec = c_puct * Pas_vec * (np.sqrt(n) * np.ones(dtype=np.float32, shape=Nas_vec.shape) / (1 + Nas_vec))
        decision_vec = Qas_vec + Uas_vec

        mask = np.reshape(np.array(node.valid_actions_mask_vec, dtype=np.bool), [9])
        idx_for_selected_action_within_space_of_valid_actions = np.argmax(np.squeeze(decision_vec, [0])[mask])

        idxs_for_all_actions_within_space_of_all_actions = np.arange(9)
        idxs_for_valid_actions_within_space_of_all_actions = idxs_for_all_actions_within_space_of_all_actions[mask]
        idx_for_selected_action_within_space_of_all_actions = idxs_for_valid_actions_within_space_of_all_actions[idx_for_selected_action_within_space_of_valid_actions]

        a = idx_for_selected_action_within_space_of_all_actions

        # action a leads to leaf node
        if node.child_nodes[a] is None:

            # compute leaf node state
            cloned_env = node.s.clone()
            cloned_env.step(move=(a // 3, a % 3), player_id=cloned_env.state.turn_parity.sign)

            # construct leaf node
            new_node = Node(s=cloned_env, parent_node_and_prev_action=(node, a))

            # attach leaf node to current node
            node.child_nodes[a] = new_node

            node = new_node
        else:
            node = node.child_nodes[a]

    # if its a leaf node that isnt a terminal state, use the neural network to fill in the value and prior
    if node.is_leaf_node and not node.is_terminal_node:
        p_vec_leaf, v_leaf = node.predict_value_and_prior_vec_for_leaf_node(sess, f_theta)
        node.Pas_vec = p_vec_leaf
        node.Vs = v_leaf

    # if it's a terminal state we've never visited before, use the actual outcome for the value
    elif node.is_terminal_node and node.Vs is None:
        v_leaf = node.get_value_for_terminal_node()
        node.Vs = v_leaf

    # if it's a terminal node whose value was computed on a previous simulation
    else:
        v_leaf = node.Vs

    sign = -1.0

    # backtrack and update the action values
    while node.has_parent_node:
        parent_node = node.parent_node
        a = node.prev_action

        N = parent_node.Nas_vec[a]

        parent_node.Nas_vec[a] += 1
        parent_node.n += 1
        parent_node.Was_vec[a] += sign * v_leaf
        parent_node.Qas_vec[a] = parent_node.Was_vec[a] / float(parent_node.Nas_vec[a])

        node = parent_node

        sign *= -1.0

    return node



def run_me_the_mtcs(tree, sess, f_theta, num_sims, t):
    # t denotes the move number of the move currently being decided.
    # when computing pi_t, the alphago zero people use tau = 1.0 early in game, and use a low temperature parameter tau for the later moves in the game.
    # for now, we will stick with tau = 1.0 the whole game, but having t as an argument in this function makes this functionality easy to change, if we so wish.

    if t <= 3:
        tau = 1.0
    else:
        tau = 0.5

    for sim in range(0,num_sims):
        tree = run_me_a_simulation(tree, sess, f_theta)

    pi_t = np.power(tree.Nas_vec, (1.0/tau)) / np.sum(np.power(tree.Nas_vec, (1.0/tau)), keepdims=True)

    return tree, pi_t

def choose_a_move(pi_t):
    a_t = np.random.choice(9, None, p=pi_t)
    return a_t

def set_new_root(tree, a_t):
    tree = tree.child_nodes[a_t]
    tree.is_root = True
    tree.parent_node = None
    tree.prev_action = None
    eps = 0.25

    #alpha = 0.03  # for go
    alpha = 1.21  # for tictactoe

    # per the deepmind alphazero paper, alpha is scaled in inverse proportion to avg num legal moves: 
    # so for go's alpha to be alpha = 0.03, we have 0.03 = c * 1/(362/2) for some c,
    # and thus c = 0.03 * (362/2), 
    # so that for tictactoe, alpha is approximately [0.03 * (362/2)] / (9/2) ~= 1.21

    num_valid_actions = np.sum(tree.valid_actions_mask_vec)
    boolean_valid_actions_mask = np.reshape(np.array(tree.valid_actions_mask_vec, dtype=np.bool), [9])

    idxs_for_all_actions_within_space_of_all_actions = np.arange(9)
    idxs_for_valid_actions_within_space_of_all_actions = idxs_for_all_actions_within_space_of_all_actions[boolean_valid_actions_mask]

    alpha_vec = alpha * np.ones(shape=(num_valid_actions,))

    dirichlet_noise_within_space_of_valid_actions = np.random.dirichlet(alpha_vec)
    dirichlet_noise_for_valid_actions_within_space_of_all_actions = np.zeros(dtype=np.float32, shape=(9,))
    dirichlet_noise_for_valid_actions_within_space_of_all_actions[idxs_for_valid_actions_within_space_of_all_actions] = dirichlet_noise_within_space_of_valid_actions

    tree.Pas_vec = eps * dirichlet_noise_for_valid_actions_within_space_of_all_actions + (1. - eps) * tree.Pas_vec
    return tree


# in progress
def run_me_a_game(sess, f_theta, num_sims=100, debug=False):
    game = TicTacToe()
    tree = get_me_a_tree(game)
    train_log = []
    game_log = []
    debug_log = []
    t = 0

    reward = None

    while True:
        s_t = game.clone()
        valid_t = tree.valid_actions_mask_vec

        tree, pi_t = run_me_the_mtcs(tree, sess, f_theta, num_sims=num_sims, t=t)

        Vs = tree.Vs
        Pas_vec = tree.Pas_vec
        Qas_vec = tree.Qas_vec
        Nas_vec = tree.Nas_vec

        a_t = choose_a_move(pi_t)
        tree = set_new_root(tree, a_t)

        train_log.append((s_t, valid_t, pi_t))
        game_log.append((s_t, a_t))

        s_tp1, reward, episode_done = game.step(move=(a_t // 3, a_t % 3), player_id=game.state.turn_parity.sign)

        if debug:
            debug_log.append({
                't': t,
                's_t': s_t, 
                'Vs': Vs, 
                'valid_t': valid_t, 
                'Pas_vec': Pas_vec, 
                'Qas_vec': Qas_vec, 
                'Nas_vec': Nas_vec, 
                'pi_t': pi_t, 
                'a_t': a_t,
                'r_t': reward,
                'episode_done': episode_done})

        t += 1

        if episode_done:
            #print('episode done')
            game_log.append((s_tp1, 'game over'))
            break

    train_log.reverse()
    train_log = [(tuple_[0], tuple_[1], tuple_[2], ((-1.0) ** i) * reward) for i, tuple_ in enumerate(train_log, 0)]
    train_log.reverse()
    return train_log, game_log, debug_log



class NeuralNetwork:
    def __init__(self):
        self.state = tf.placeholder(tf.int32, [None, 3, 3])
        self.valid_actions_mask = tf.placeholder(tf.int32, [None, 9])
        self.pi = tf.placeholder(tf.float32, [None, 9])
        self.z = tf.placeholder(tf.float32, [None, 1])
        self.is_train = tf.placeholder(tf.bool, [])
        self.lr = tf.placeholder(tf.float32, [])

        with tf.variable_scope('nn'):
            board = tf.expand_dims(tf.cast(self.state, dtype=tf.float32), -1)

            channel_dim = 40

            ## residual tower
            # conv block
            conv1 = tf.layers.conv2d(board, filters=channel_dim, kernel_size=[2,2], strides=[1,1], padding='VALID', activation=None)
            bn1 = tf.layers.batch_normalization(conv1, training=self.is_train)
            relu1 = tf.nn.relu(bn1)

            # residual block 1
            conv2 = tf.layers.conv2d(relu1, filters=channel_dim, kernel_size=[2,2], strides=[1,1], padding='SAME', activation=None)
            bn2 = tf.layers.batch_normalization(conv2, training=self.is_train)
            relu2 = tf.nn.relu(bn2)
            conv3 = tf.layers.conv2d(relu2, filters=channel_dim, kernel_size=[2,2], strides=[1,1], padding='SAME', activation=None)
            bn3 = tf.layers.batch_normalization(conv3, training=self.is_train)
            res3 = relu1 + bn3
            relu3 = tf.nn.relu(res3)

            # residual block 2
            conv4 = tf.layers.conv2d(relu3, filters=channel_dim, kernel_size=[2,2], strides=[1,1], padding='SAME', activation=None)
            bn4 = tf.layers.batch_normalization(conv4, training=self.is_train)
            relu4 = tf.nn.relu(bn4)
            conv5 = tf.layers.conv2d(relu4, filters=channel_dim, kernel_size=[2,2], strides=[1,1], padding='SAME', activation=None)
            bn5 = tf.layers.batch_normalization(conv5, training=self.is_train)
            res5 = relu3 + bn5
            relu5 = tf.nn.relu(res5)

            # residual block 3
            conv6 = tf.layers.conv2d(relu5, filters=channel_dim, kernel_size=[2,2], strides=[1,1], padding='SAME', activation=None)
            bn6 = tf.layers.batch_normalization(conv6, training=self.is_train)
            relu6 = tf.nn.relu(bn6)
            conv7 = tf.layers.conv2d(relu6, filters=channel_dim, kernel_size=[2,2], strides=[1,1], padding='SAME', activation=None)
            bn7 = tf.layers.batch_normalization(conv7, training=self.is_train)
            res7 = relu5 + bn7
            relu7 = tf.nn.relu(res7)

            shared_features = relu7

            ## policy head
            policy_conv = tf.layers.conv2d(shared_features, filters=2, kernel_size=[1,1], strides=[1,1], padding='VALID', activation=None)
            policy_bn = tf.layers.batch_normalization(policy_conv, training=self.is_train)
            policy_relu = tf.nn.relu(policy_bn)
            policy_flatten = tf.reshape(policy_relu, [-1, 3*3*2])
            policy_fc = tf.layers.dense(policy_flatten, 9, activation=None)
            logits = policy_fc

            softmax_terms = tf.cast(self.valid_actions_mask, dtype=tf.float32) * tf.exp(tf.reshape(logits, [-1, 9]))
            self.probabilities = softmax_terms / tf.reduce_sum(softmax_terms, axis=1, keep_dims=True)

            ## value head
            value_conv = tf.layers.conv2d(shared_features, filters=1, kernel_size=[1,1], strides=[1,1], padding='VALID', activation=None)
            value_bn = tf.layers.batch_normalization(value_conv, training=self.is_train)
            value_relu1 = tf.nn.relu(value_bn)
            value_flatten = tf.reshape(value_relu1, [-1, 3*3*1])
            value_fc1 = tf.layers.dense(value_flatten, channel_dim, activation=None)
            value_relu2 = tf.nn.relu(value_fc1)
            value_fc2 = tf.layers.dense(value_relu2, 1, activation=None)
            value_tanh = tf.nn.tanh(value_fc2)

            self.value = value_tanh

            self.loss_terms = tf.squeeze(tf.square(self.z - self.value), [1]) - tf.reduce_sum(self.pi * tf.log(self.probabilities + 1e-8), axis=1)
            self.loss = tf.reduce_mean(self.loss_terms, axis=0)
            self.optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, state, valid_actions_mask):
        feed_dict = {
            self.state: state,
            self.valid_actions_mask: valid_actions_mask,
            self.is_train: False
        }
        p_theta, v_theta = sess.run([self.probabilities, self.value], feed_dict=feed_dict)
        return p_theta, v_theta

    def train(self, sess, state, valid_actions_mask, pi, z, lr):
        feed_dict = {
            self.state: state,
            self.valid_actions_mask: valid_actions_mask,
            self.pi: pi,
            self.z: z,
            self.is_train: True,
            self.lr: lr
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss


f_theta = NeuralNetwork()

game.reset()

init_op = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init_op)

# tree = get_me_a_tree(game)
# tree, pi_t = run_me_the_mtcs(tree, sess, f_theta, num_sims=1000)

train_log_before, game_log_before, debug_log_before = run_me_a_game(sess, f_theta, num_sims=100, debug=True)


def training_step(sess, f_theta, lr, num_games=256, batch_size=64, num_sims=100, gradient_steps=8, verbose=False):
    train_log_combined = []
    for _ in range(0, num_games):
        train_log, _, _ = run_me_a_game(sess, f_theta, num_sims=num_sims)
        train_log_combined.extend(train_log)

    shuffle(train_log_combined)

    for i in range(0, (len(train_log_combined) - (len(train_log_combined) % batch_size)) // batch_size):
        state_batch = []
        valid_actions_mask_batch = []
        pi_batch = []
        z_batch = []

        for j in range(0, batch_size):
            if verbose:
                print('i,j: {},{}'.format(i, j))
            row = train_log_combined[batch_size * i + j]
            s_t, valid_t, pi_t, z_t = row[0], row[1], row[2], row[3]

            state_repr_t = s_t.state.board * s_t.state.turn_parity.sign
      
            state_batch.append(state_repr_t)
            valid_actions_mask_batch.append(valid_t)
            pi_batch.append(pi_t)
            z_batch.append([z_t])

        state_batch = np.array(state_batch)
        valid_actions_mask_batch = np.array(valid_actions_mask_batch)
        pi_batch = np.array(pi_batch)
        z_batch = np.array(z_batch)

        loss_batch = f_theta.train(sess, state_batch, valid_actions_mask_batch, pi_batch, z_batch, lr)

        print('loss: {}'.format(loss_batch))

        if i == gradient_steps - 1:
            break


def training_loop(sess, f_theta, mode='basic'):
    settings = {
        'debug': {
            'steps': [1, 1, 1],
            'games': 64,
            'sims': 20
        },
        'basic': {
            'steps': [20, 10, 10],
            'games': 256,
            'sims': 100
        },
        'standard': {
            'steps': [100, 50, 50],
            'games': 256,
            'sims': 100
        }
    }
    i1, i2, i3 = settings[mode]['steps']
    num_games = settings[mode]['games']
    num_sims = settings[mode]['sims']

    for _ in range(0, i1):
        training_step(sess, f_theta, lr=0.01, num_games=num_games, num_sims=num_sims)
    for _ in range(0, i2):
        training_step(sess, f_theta, lr=0.001, num_games=num_games, num_sims=num_sims)
    for _ in range(0, i3):
        training_step(sess, f_theta, lr=0.0001, num_games=num_games, num_sims=num_sims)





def play_me_a_game(sess, f_theta, num_sims=100, human_player_id=None):
    assert human_player_id is None or human_player_id in [-1, 1]
    if human_player_id is None:
        coin_toss = np.random.randint(2)
        called_it = input("coin toss for the first move. call it 0 or 1.\n> ")
        time.sleep(3)
        print('\n')

        if int(called_it) == coin_toss:
            human_player_id = 1
            remark = ', you called it right!'
        else:
            human_player_id = -1
            remark = '...'
        
        print('coin toss was {}{}\n'.format(coin_toss, remark))
        print('you play as {}'.format(human_player_id))
        time.sleep(2)

    print('player 1 goes first.')
    time.sleep(1)

    game = TicTacToe()
    tree = get_me_a_tree(game)

    episode_done = False

    t = 0

    while not episode_done:

        if game.state.turn_parity.sign == human_player_id:
            print('_____________\nnew turn.\n')
            print(game)
            move = input("\nit's your turn. where would you like to go? enter a tuple (i,j)\n> ")
            move = move.strip()
            time.sleep(2)

            if move == 'q' or move == 'quit' or move == 'Q':
                print('human has quit the game.')
                break

            if len(move) == 0:
                print('input does not conform with required format for a move.')
                continue

            if move[0] != '(' or move[-1] != ')':
                print('input does not conform with required format for a move.')
                continue

            coords = move[1:-1].strip('\n').strip(' ').split(',')
            coords = [str(s.strip(' ')) for s in coords]
            if len(coords) != 2:
                print('input does not conform with required format for a move.')
                continue

            coord0 = coords[0]
            coord1 = coords[1]
            bool0 = coord0 in [str(i) for i in range(0,3)]
            bool1 = coord1 in [str(i) for i in range(0,3)]
            if not bool0 or not bool1:
                print('input coordinates not in valid coordinate set.')
                continue

            move = (int(coord0), int(coord1))

            if not game.move_is_valid(move=move, player_id=game.state.turn_parity.sign):
                print('move is not valid!')
                continue

            # in current implementation, the bot is gonna run some simulations every turn, even if it's the human's move.
            # because, why not increase our visitation counts?
            tree, pi_t = run_me_the_mtcs(tree, sess, f_theta, num_sims=num_sims, t=t)
            a_t = 3 * move[0] + move[1]
            tree = set_new_root(tree, a_t)

        else:
            print('_____________\nnew turn.\n')
            print(game)
            print("\nit's the bot's turn. where will it go?")

            tree, pi_t = run_me_the_mtcs(tree, sess, f_theta, num_sims=num_sims, t=t)
            a_t = choose_a_move(pi_t)
            tree = set_new_root(tree, a_t)

            print('the bot played at ({},{}).\n'.format(a_t // 3, a_t % 3))
            time.sleep(3)

        t += 1

        s_tp1, reward, episode_done = game.step(move=(a_t // 3, a_t % 3), player_id=game.state.turn_parity.sign)
        if episode_done:
            time.sleep(3)
            print('_____________\ngame over.\n')
            print(game)
            print('\n')
            print('thanks for the game.')
            print('good game.')
            print('\n')
            break


mode_selector = 'basic'
training_loop(sess, f_theta, mode=mode_selector)

train_log_after, game_log_after, debug_log_after = run_me_a_game(sess, f_theta, num_sims=100, debug=False)


#play_me_a_game(sess, f_theta, num_sims=100)
