# # Base Data Science snippet
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist
import imageio
import sys

# Tambahkan path ke folder rl
sys.path.append("../")
from rl.agents.q_agent import QAgent

# Mengatur style plot
plt.style.use("classic")


class DeliveryEnvironment(object):
    """
    Lingkungan TSP yang disederhanakan, hanya berfokus pada optimasi jarak.
    """
    def __init__(self, n_stops=10, max_box=10):
        print(f"Initialized Delivery Environment with {n_stops} random stops.")
        print("Target metric for optimization is distance.")

        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.max_box = max_box
        
        self._generate_stops()
        self._generate_q_values()
        self.reset()

    def _generate_stops(self):
        """Menghasilkan koordinat geografis acak untuk setiap titik pemberhentian."""
        xy = np.random.rand(self.n_stops, 2) * self.max_box
        self.x = xy[:, 0]
        self.y = xy[:, 1]

    def _generate_q_values(self):
        """Menghasilkan matriks biaya berdasarkan jarak Euklides antar titik."""
        xy = np.column_stack([self.x, self.y])
        self.q_stops = cdist(xy, xy)

    def render(self, return_img=False):
        """Memvisualisasikan titik-titik dan rute yang diambil."""
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title(f"Delivery Stops (Route: {len(self.stops)}/{self.n_stops})")

        ax.scatter(self.x, self.y, c="red", s=50)

        if len(self.stops) > 0:
            start_xy = (self.x[self.stops[0]], self.y[self.stops[0]])
            ax.annotate("START", xy=start_xy, xytext=(start_xy[0] + 0.1, start_xy[1] - 0.05), weight="bold")

        if len(self.stops) > 1:
            route_coords = np.array([(self.x[stop], self.y[stop]) for stop in self.stops])
            ax.plot(route_coords[:, 0], route_coords[:, 1], c="blue", linewidth=1, linestyle="--")
            
            if len(self.stops) == self.n_stops:
                end_xy = (self.x[self.stops[-1]], self.y[self.stops[-1]])
                ax.annotate("END", xy=end_xy, xytext=(end_xy[0] + 0.1, end_xy[1] - 0.05), weight="bold")

        plt.xticks([])
        plt.yticks([])

        if return_img:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            return image
        else:
            plt.show()

    def reset(self):
        """Mereset lingkungan untuk episode baru."""
        self.stops = []
        first_stop = np.random.randint(self.n_stops)
        self.stops.append(first_stop)
        return first_stop

    def step(self, destination):
        """Menjalankan satu langkah, menghitung reward yang benar."""
        state = self.stops[-1]
        
        if destination in self.stops:
            reward = -100  # Penalti besar untuk langkah ilegal
        else:
            reward = -self.q_stops[state, destination]
            self.stops.append(destination)
        
        done = len(self.stops) == self.n_stops
        return destination, reward, done

class DeliveryQAgent(QAgent):
    """Agen Q-Learning yang diadaptasi untuk TSP."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_memory()

    def act(self, s):
        """Pilih aksi, hindari state yang sudah dikunjungi."""
        q = np.copy(self.Q[s, :])
        q[self.states_memory] = -np.inf
        
        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            possible_actions = [x for x in range(self.actions_size) if x not in self.states_memory]
            if not possible_actions: return np.random.randint(self.actions_size)
            a = np.random.choice(possible_actions)
        return a

    def remember_state(self, s):
        """Mengingat state yang telah dikunjungi dalam episode ini."""
        self.states_memory.append(s)

    def reset_memory(self):
        """Mereset memori kunjungan untuk episode baru."""
        self.states_memory = []

    def find_optimal_route(self, env):
        """
        Gunakan Q-Table yang sudah dilatih untuk menemukan rute terbaik secara deterministik.
        Waktu eksekusi fungsi ini adalah 'waktu komputasi' yang akan dibandingkan.
        """
        start_time = time.time()
        
        s = env.reset()
        self.reset_memory()
        route = [s]
        total_distance = 0.0
        
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # 100% eksploitasi
        
        done = False
        while not done:
            self.remember_state(s)
            a = self.act(s)
            total_distance += env.q_stops[s, a]
            s_next, _, done = env.step(a)
            s = s_next
            if not done: route.append(s)

        self.epsilon = original_epsilon
        inference_time = time.time() - start_time
        
        return route, total_distance, inference_time

def run_training_with_visualization(env, agent, n_episodes=1000, gif_filename="training_visual.gif", render_each=10,fps=10):
    """
    Menjalankan loop pelatihan, menyimpan gambar untuk GIF, DAN mencatat riwayat reward.
    """
    print(f"Memulai pelatihan untuk {n_episodes} episode...")
    start_time = time.time()
    images = []
    
    # BARU: Inisialisasi list untuk menyimpan riwayat reward
    rewards_history = []
    
    for i in tqdm(range(n_episodes)):
        s = env.reset()
        agent.reset_memory()
        
        # BARU: Inisialisasi reward untuk episode saat ini
        episode_reward = 0
        done = False
        
        while not done:
            agent.remember_state(s)
            a = agent.act(s)
            s_next, r, done = env.step(a)
            agent.train(s, a, r, s_next)
            
            # BARU: Akumulasi reward untuk episode ini
            episode_reward += r
            s = s_next
        
        # BARU: Simpan total reward dari episode yang baru selesai
        rewards_history.append(episode_reward)
        
        # Simpan gambar untuk GIF secara berkala
        if i % render_each == 0:
            # Render rute terakhir dari episode ini untuk divisualisasikan
            images.append(env.render(return_img=True))

    total_training_time = time.time() - start_time
    print(f"\nPelatihan selesai dalam {total_training_time:.4f} detik.")
    
    # Simpan GIF
    print(f"Menyimpan visualisasi pelatihan sebagai '{gif_filename}'...")
    imageio.mimsave(gif_filename, images, fps=fps)
    print("Visualisasi berhasil disimpan.")
    
    # BARU: Kembalikan juga riwayat reward
    return agent, total_training_time, rewards_history