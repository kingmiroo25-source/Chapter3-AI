# Example 3.23: OpenAI Gym CartPole (Updated for Gymnasium)
# Documentation: https://gymnasium.farama.org/environments/classic_control/cartpole/

import gymnasium as gym

# 1. Inisialisasi environment dengan render_mode='human' agar animasi visual muncul
env = gym.make("CartPole-v1", render_mode="human")

# 2. Perulangan untuk 20 episode
for i_episode in range(20):
    # Gymnasium reset() mengembalikan tuple (observation, info)
    observation, info = env.reset()

    for t in range(100):
        print(observation)

        # Pilih aksi secara acak (0: kiri, 1: kanan)
        action = env.action_space.sample()

        # Gymnasium step() mengembalikan 5 nilai
        observation, reward, terminated, truncated, info = env.step(action)

        # Cek apakah tiang jatuh (terminated) atau batas langkah habis (truncated)
        if terminated or truncated:
            print(f"Episode {i_episode + 1} finished after {t + 1} timesteps")
            break

# Tutup jendela simulasi setelah selesai
env.close()   
