import numpy as np

from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def save_data(agent, name, scores):
    now = datetime.now().strftime('%Y-%m-%d_%H%M')
    train_name = 'train_{}_{}'.format(name, now)
    train_dir = Path(train_name)
    train_dir.mkdir(exist_ok=True)

    actor_local_path = train_dir / 'checkpoint_actor_local_{}.pth'.format(train_name)
    actor_target_path = train_dir / 'checkpoint_actor_target_{}.pth'.format(train_name)
    critic_local_path = train_dir / 'checkpoint_critic_local_{}.pth'.format(train_name)
    critic_target_path = train_dir / 'checkpoint_critic_target_{}.pth'.format(train_name)
    agent.save(actor_local_path, actor_target_path,
               critic_local_path, critic_target_path)

    logfile = train_dir / "score_{}.csv".format(train_name)
    rows = ["{},{}".format(i+1, v) for i, v in enumerate(scores)]
    score_csv = ["episode,score"] + rows
    with logfile.open(mode='w') as f:
        f.write('\n'.join(score_csv))


def train(env, agent, n_episodes=200, max_t=1000):
    # get the default brain
    brain_name = env.brain_names[0]

    train_id = str(uuid4())[:8]

    scores = []
    scores_window = deque(maxlen=100)
    elapses = []

    once_solved = False

    time_train_start = datetime.now()

    for i_episode in range(1, n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        # state = env_info.vector_observations[0]
        states = env_info.vector_observations
        agent.reset()

        # score = 0
        episode_scores = np.zeros(len(env_info.agents))

        # to measure time elapsed
        time_episode_start = datetime.now()

        # run the episode
        for t in range(max_t):
            # action = agent.act(state)
            actions = agent.act(states)  # multiple

            # send the action to the environment
            # env_info = env.step(action)[brain_name]
            env_info = env.step(actions)[brain_name]  # multiple

            # next_state = env_info.vector_observations[0]
            # reward = env_info.rewards[0]
            # done = env_info.local_done[0]
            next_states = env_info.vector_observations  # multiple
            rewards = env_info.rewards
            dones = env_info.local_done

            # agent.step(state, action, reward, next_state, done)
            experiences = zip(states, actions, rewards, next_states, dones)
            for state, action, reward, next_state, done in experiences:
                agent.step(state, action, reward, next_state, done)

            # score += reward
            # state = next_state
            # if done:
            #     break
            episode_scores += rewards
            states = next_states
            if np.any(dones):
                break


        time_episode_end = datetime.now()

        # scores_window.append(score)
        # scores.append(score)
        m_scores = np.mean(episode_scores)
        scores_window.append(m_scores)
        scores.append(m_scores)

        episode_elapsed = time_episode_end - time_episode_start
        elapses.append(episode_elapsed)

        msg = '\rEpisode {}\tAverage Score: {:.2f}\tTime: {}'
        print(msg.format(i_episode, np.mean(scores_window), episode_elapsed), end="")

        if i_episode % 20 == 0:
            total_elapsed = time_episode_end - time_train_start
            current_mean = np.mean(scores_window)
            print(msg.format(i_episode, current_mean, total_elapsed))
            if current_mean > 8.0:
                name = '{}_ep_{}'.format(train_id, i_episode)
                save_data(agent, name, scores)


        if not once_solved and i_episode > 100 and np.mean(scores_window) >= 30.0:
            total_elapsed = time_episode_end - time_train_start
            msg_solved = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTotal Time: {}'
            print(msg_solved.format(i_episode-100, np.mean(scores_window), total_elapsed))

            name = '{}_solved'.format(train_id)
            save_data(agent, name, scores)
            # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

            once_solved = True

    return scores, elapses
