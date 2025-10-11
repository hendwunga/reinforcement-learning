import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ============================================================
# 🌆 ENVIRONMENT UNTUK DELIVERY BERDASARKAN JARAK SAJA
# ============================================================
class DeliveryEnvironment:
    def __init__(self, n_stops=10, max_box=10):
        """
        Environment sederhana untuk kasus delivery dengan reward berbasis jarak.
        - n_stops: jumlah titik pengantaran
        - max_box: ukuran peta kota (area)
        """
        self.n_stops = n_stops
        self.max_box = max_box
        self.action_space = n_stops
        self.observation_space = n_stops

        # Buat titik-titik pengantaran secara acak di peta
        self._generate_stops()

        # Hitung jarak antar semua titik (Euclidean distance)
        self._generate_q_values()

        # Set posisi awal agent
        self.reset()

    # ============================================================
    # Membuat titik-titik pengantaran acak di area kota
    # ============================================================
    def _generate_stops(self):
        xy = np.random.rand(self.n_stops, 2) * self.max_box
        self.x = xy[:, 0]
        self.y = xy[:, 1]

    # ============================================================
    # Hitung matriks jarak antar titik
    # ============================================================
    def _generate_q_values(self):
        xy = np.column_stack([self.x, self.y])
        self.q_stops = cdist(xy, xy)  # Matriks jarak Euclidean

    # ============================================================
    # Fungsi reset environment untuk memulai episode baru
    # ============================================================
    def reset(self):
        self.visited = set()  # Titik-titik yang sudah dikunjungi
        self.current_state = np.random.randint(self.n_stops)
        self.visited.add(self.current_state)
        return self.current_state

    # ============================================================
    # Fungsi step() → Agent bergerak ke titik berikutnya
    # ============================================================
    def step(self, action):
        if action in self.visited:
            # Jika titik sudah dikunjungi → penalti besar
            reward = -10
            done = False
        else:
            # Hitung jarak antar titik (reward = jarak)
            reward = self.q_stops[self.current_state, action]

            # Perbarui posisi dan daftar titik yang dikunjungi
            self.current_state = action
            self.visited.add(action)

            # Cek apakah semua titik sudah dikunjungi
            done = len(self.visited) == self.n_stops

        return self.current_state, reward, done

    # ============================================================
    # Fungsi render → Gambar peta titik dan rute pengantaran
    # ============================================================
    def render(self):
        plt.figure(figsize=(6,6))
        plt.scatter(self.x, self.y, c='blue', label='Stops')
        plt.scatter(self.x[self.current_state], self.y[self.current_state], c='red', label='Current Position')
        plt.legend()
        plt.title("Delivery Environment (Distance-based)")
        plt.show()
