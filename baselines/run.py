import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env
from baselines.common.tf_util import get_session
from baselines import bench, logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common import atari_wrappers, retro_wrappers


# Side-load our traffic project ########################################################################################
import sys
import os
sys.path.append(os.path.abspath('/home/atcold/Work/GitHub/pytorch-Traffic-Simulator'))
########################################################################################################################

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

_game_envs['traffic'] = {'I-80-v1'}


def train(args, extra_args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)
    args.__dict__.update(extra_args)

    env = build_env(args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if 'I-80-v1' not in gym.envs.registry.env_specs:
        gym.envs.registration.register(
            id='I-80-v1',
            entry_point='map_i80_ctrl:ControlledI80',
            kwargs=dict(
                fps=50,
                nb_states=args.n_cond,
                display=args.play,
                delta_t=0.1,
                return_reward=True,
                normalise_state=True,
                normalise_action=True,
                gamma=float(args.gamma),
                show_frame_count=False,
            ),
        )

    if env_type == 'atari':
        if alg == 'acer':
            env = make_vec_env(env_id, env_type, nenv, seed)
        elif alg == 'deepq':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir())
            env = atari_wrappers.wrap_deepmind(env, frame_stack=True, scale=True)
        elif alg == 'trpo_mpi':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            env = atari_wrappers.wrap_deepmind(env)
            # TODO check if the second seeding is necessary, and eventually remove
            env.seed(seed)
        else:
            frame_stack_size = 4
            env = VecFrameStack(make_vec_env(env_id, env_type, nenv, seed), frame_stack_size)

    elif env_type == 'retro':
        import retro
        gamestate = args.gamestate or 'Level1-1'
        env = retro_wrappers.make_retro(game=args.env, state=gamestate, max_episode_steps=10000,
                                        use_restricted_actions=retro.Actions.DISCRETE)
        env.seed(args.seed)
        env = bench.Monitor(env, logger.get_dir())
        env = retro_wrappers.wrap_deepmind_retro(env)

    else:
        get_session(tf.ConfigProto(allow_soft_placement=True,
                                    intra_op_parallelism_threads=1,
                                    inter_op_parallelism_threads=1))

        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale)
        # env = make_vec_env(env_id, env_type, nenv, seed, reward_scale=args.reward_scale)

        if env_type == 'mujoco':
            env = VecNormalize(env)

        if env_type == 'traffic':
            env = VecNormalize(env, ob=False, ret=True)

    return env


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type == 'atari':
        return 'cnn'
    if env_type == 'traffic':
        return 'traffic_model'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()
        def initialize_placeholders(nlstm=128,**kwargs):
            return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
        state, dones = initialize_placeholders(**extra_args)
        while True:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
            obs, _, done, _ = env.step(actions)
            env.render()
            # done = done.any() if isinstance(done, np.ndarray) else done

            # dummy_vec_env is resetting the env already
            # if done:
            #     obs = env.reset()

    if args.eval:
        print('Evaluating PPO')
        import pickle
        from tqdm import tqdm

        file_path = '/home/atcold/Work/GitHub/pytorch-Traffic-Simulator'
        with open(f'{file_path}/test_indx.pkl', 'rb') as f:
            test_data = pickle.load(f)

        TIMESLOT = 0
        CAR_ID = 1

        env = build_env(args)
        distances = list()
        arrived = list()
        for k, v in tqdm(test_data.items()):
            # print(f'Processing sample {k}: car {v[CAR_ID]} from timeslot {v[TIMESLOT]}')
            obs = env.reset(time_slot=v[TIMESLOT], vehicle_id=v[CAR_ID])  # if None => picked at random
            car = env.venv.envs[0].env.controlled_car['locked']
            initial_position = car._position[0]
            done = False
            while not done:
                actions, _, _, _ = model.step(obs)
                obs, _, done, _ = env.step(actions)
                done = done.any()
                # env.render()
            final_position = car._position[0]
            travelled_distance = final_position - initial_position
            distances.append(travelled_distance)
            arrived.append(car.arrived_to_dst)

        print('Dumping stats to file')
        os.system('mkdir -p PPO-performance')
        with open(f'PPO-performance/{osp.basename(extra_args.get("load_path"))}.pkl', 'wb') as f:
            pickle.dump(dict(
                distances=distances,
                arrived=arrived
            ), f)

        env.close()

if __name__ == '__main__':
    main()
