import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import imageio
from tqdm import tqdm

# Mengatur gaya visual default
plt.style.use("seaborn-v0_8-darkgrid")
import sys
sys.path.append("../") 
from rl.agents.q_agent import QAgent

class TspTimeEnvironment(object):
    """
    Kelas Environment khusus untuk TSP di mana metrik optimasinya adalah
    WAKTU TEMPUH yang stokastik (berubah-ubah/acak).
    """
    def __init__(self, n_stops=10, world_size=10, avg_speed=1.0, traffic_variability=0.3):
        print(f"Membuat Environment TSP Waktu dengan {n_stops} titik acak.")
        self.n_stops = n_stops
        self.world_size = world_size
        self.traffic_variability = traffic_variability
        self.action_space = n_stops
        self.observation_space = n_stops
        
        self.x, self.y, self.stops = None, None, []
        
        self._generate_stops()
        self._generate_distance_matrix()
        self._generate_base_time_matrix(avg_speed)
        
        self.reset()

    def _generate_stops(self):
        xy = np.random.rand(self.n_stops, 2) * self.world_size
        self.x, self.y = xy[:, 0], xy[:, 1]

    def _generate_distance_matrix(self):
        xy = np.column_stack([self.x, self.y])
        self.distance_matrix = cdist(xy, xy)
        
    def _generate_base_time_matrix(self, avg_speed):
        if avg_speed <= 0: avg_speed = 1.0
        self.base_time_matrix = self.distance_matrix / avg_speed

    # --- INI ADALAH PERBEDAAN UTAMA DENGAN VERSI JARAK ---
    def _get_reward(self, state, new_state):
        # Ambil waktu tempuh rata-rata (ideal)
        base_time = self.base_time_matrix[state, new_state]
        
        # Tambahkan faktor acak untuk mensimulasikan lalu lintas yang tidak terduga
        # 'randn()' menghasilkan angka acak (bisa positif/negatif) di sekitar nol
        traffic_noise = base_time * self.traffic_variability * np.random.randn()
        
        # Waktu tempuh aktual adalah waktu ideal + gangguan lalu lintas
        # Gunakan max(0, ...) untuk memastikan waktu tidak pernah negatif
        actual_travel_time = max(0, base_time + traffic_noise)
        
        # Reward adalah waktu tempuh aktual. Agen akan belajar meminimalkan "biaya" ini.
        return actual_travel_time

    def reset(self):
        self.stops = []
        first_stop = np.random.randint(self.n_stops)
        self.stops.append(first_stop)
        return first_stop

    def step(self, destination):
        state = self._get_state()
        reward = self._get_reward(state, destination)
        self.stops.append(destination)
        done = len(self.stops) == self.n_stops
        return destination, reward, done
    
    def _get_state(self):
        return self.stops[-1]

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_state()
        return self.x[state], self.y[state]

    def render(self, return_img=False):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("TSP Stops (Time-Based Problem)", fontsize=14, fontweight='bold')
        ax.scatter(self.x, self.y, c="#e74c3c", s=60, label="Stops")
        if self.stops:
            sx, sy = self._get_xy(initial=True)
            ax.annotate("START", xy=(sx, sy), xytext=(sx + 0.1*self.world_size/10, sy - 0.05*self.world_size/10),
                        fontweight="bold", color="#2c3e50")
        if len(self.stops) > 1:
            ax.plot(self.x[self.stops], self.y[self.stops], c="#2980b9", linewidth=2, linestyle="--")
            ex, ey = self._get_xy(initial=False)
            ax.annotate("END", xy=(ex, ey), xytext=(ex + 0.1*self.world_size/10, ey - 0.05*self.world_size/10),
                        fontweight="bold", color="#2c3e50")
        ax.set_xticks([]); ax.set_yticks([]); ax.legend()
        if return_img:
            fig.canvas.draw(); image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,)); plt.close(fig); return image
        else:
            plt.show()

# =========================================================================
# ==      KELAS: Agen Spesialis TSP (Tidak Perlu Diubah)                  ==
# =========================================================================
class DeliveryQAgent(QAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_memory()

    def act(self, s):
        q = np.copy(self.Q[s, :])
        q[self.states_memory] = -np.inf
        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_memory])
        return a

    def remember_state(self, s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []

# =========================================================================
# ==      FUNGSI PELATIHAN                      ==
# =========================================================================
def run_episode(env, agent, verbose=0):
    s = env.reset()
    agent.reset_memory()
    episode_reward = 0
    for _ in range(env.n_stops):
        agent.remember_state(s)
        a = agent.act(s)
        s_next, r, done = env.step(a)
        r = -1 * r
        agent.train(s, a, r, s_next)
        episode_reward += r
        s = s_next
        if done: break
    return env, agent, episode_reward

def run_n_episodes(env, agent, name="tsp_time_training.gif", n_episodes=1000, render_each=10):
    rewards = []
    imgs = []
    print(f"Memulai pelatihan untuk {n_episodes} episode...")
    for i in tqdm(range(n_episodes), desc="Training Progress"):
        _, _, episode_reward = run_episode(env, agent, verbose=0)
        rewards.append(episode_reward)
        if i % render_each == 0:
            imgs.append(env.render(return_img=True))
    plt.figure(figsize=(15, 3))
    plt.title("Perkembangan Reward Selama Pelatihan (Basis Waktu)"); plt.xlabel("Episode")
    plt.ylabel("Total Reward (Waktu Negatif)"); plt.plot(rewards); plt.grid(True); plt.show()
    print("Menyimpan animasi pelatihan sebagai GIF..."); imageio.mimsave(name, imgs, fps=10)
    return env, agent