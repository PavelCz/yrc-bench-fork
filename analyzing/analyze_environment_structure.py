from analyzing.utils import create_env
import numpy as np

def main():
    random_percent = 50
    env = create_env(random_percent=random_percent, start_level=0, num_levels=1000)

    obs = env.reset()

    done = False
    obs_list = []
    info_list = []
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(np.array([action]))
        # It looks like info[0]['randomize_goal'] is 1 or 0 depending on whether the coin
        # is random or not.
        obs_list.append(obs)
        info_list.append(info)

        if info[0]['randomize_goal'] == 1:
            print("Random coin")
        else:
            print("Deterministic coin")



if __name__ == "__main__":
    main()
