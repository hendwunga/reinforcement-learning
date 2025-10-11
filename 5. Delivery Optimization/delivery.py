# Base Data Science snippet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook
from scipy.spatial.distance import cdist
import imageio
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

plt.style.use("seaborn-dark")

import sys
sys.path.append("../")
from rl.agents.q_agent import QAgent



# Peta Kota ( Environment )
class DeliveryEnvironment(object):
    def __init__(self, n_stops=10, max_box=10, method="distance", **kwargs):
        print(f"Initialized Delivery Environment with {n_stops} random stops")
        print(f"Target metric for optimization is {method}")

        self.n_stops = n_stops                    # Jumlah titik yang harus dikunjungi
        self.action_space = self.n_stops          # Banyaknya aksi yang bisa diambil (sama seperti jumlah titik)
        self.observation_space = self.n_stops     # Banyaknya state (kota/titik yang bisa dikunjungi)
        self.max_box = max_box                    # Ukuran area kota (misal 10x10)
        self.stops = []                           # Menyimpan urutan titik yang sudah dikunjungi
        self.method = method                      # Menentukan mode perhitungan: distance, time, atau traffic_box

        # Generate area dan titik
        self._generate_constraints(**kwargs)      # Buat zona macet (kalau method=traffic_box)
        self._generate_stops()                    # Buat titik-titik pengantaran (koordinat acak)
        self._generate_q_values()                 # Hitung matriks jarak/waktu antar titik
        self.render()                             # Gambarkan peta kota

        self.reset()                              # Reset posisi awal (mulai dari titik acak)

    # -------------------------------------------------------------------------
    # Bagian untuk membuat kotak zona macet (hanya aktif jika method = traffic_box)
    # -------------------------------------------------------------------------
    def _generate_constraints(self, box_size=0.2, traffic_intensity=5):
        if self.method == "traffic_box":
            # Tentukan posisi acak untuk sisi kiri dan bawah dari kotak macet
            x_left = np.random.rand() * (self.max_box) * (1 - box_size)
            y_bottom = np.random.rand() * (self.max_box) * (1 - box_size)

            # Tentukan sisi kanan dan atas kotak berdasarkan ukuran box_size
            x_right = x_left + np.random.rand() * box_size * self.max_box
            y_top = y_bottom + np.random.rand() * box_size * self.max_box

            # Simpan koordinat kotak macet (x kiri, x kanan, y bawah, y atas)
            self.box = (x_left, x_right, y_bottom, y_top)
            self.traffic_intensity = traffic_intensity   # Seberapa parah efek macetnya

    # -------------------------------------------------------------------------
    # Membuat titik-titik pengantaran secara acak di dalam peta kota
    # -------------------------------------------------------------------------
    def _generate_stops(self):
        if self.method == "traffic_box":
            # Kalau ada zona macet, titik pengantaran diusahakan tidak di dalam kotak macet
            points = []
            while len(points) < self.n_stops:
                x, y = np.random.rand(2) * self.max_box
                if not self._is_in_box(x, y, self.box):  # Hindari titik di area macet
                    points.append((x, y))
            xy = np.array(points)
        else:
            # Kalau bukan mode traffic_box, semua titik acak saja di seluruh area
            xy = np.random.rand(self.n_stops, 2) * self.max_box

        # Pisahkan menjadi koordinat X dan Y
        self.x = xy[:, 0]
        self.y = xy[:, 1]

    # -------------------------------------------------------------------------
    # Membuat matriks jarak atau waktu antar titik (Q-value dasar)
    # -------------------------------------------------------------------------
    def _generate_q_values(self, box_size=0.2):
        # Kalau mode distance atau traffic_box, pakai jarak Euclidean
        if self.method in ["distance", "traffic_box"]:
            xy = np.column_stack([self.x, self.y])   # Gabungkan x dan y jadi array 2 kolom
            self.q_stops = cdist(xy, xy)             # Hitung jarak Euclidean antar semua titik

        # Kalau mode time, buat matriks waktu tempuh acak (simulasi kondisi lalu lintas)
        elif self.method == "time":
            self.q_stops = np.random.rand(self.n_stops, self.n_stops) * self.max_box
            np.fill_diagonal(self.q_stops, 0)        # Waktu dari titik ke dirinya sendiri = 0

        else:
            raise Exception("Method not recognized") # Kalau method tidak dikenal, munculkan error

    # -------------------------------------------------------------------------
    # Hitung nilai reward (biaya perjalanan) antara dua titik
    # -------------------------------------------------------------------------
    def _get_reward(self, state, new_state):
        base_reward = self.q_stops[state, new_state]  # Ambil jarak atau waktu dasar antara dua titik

        # -------------------------- MODE DISTANCE --------------------------
        if self.method == "distance":
            return base_reward                         # Reward = jarak antar titik (semakin kecil, semakin baik)

        # -------------------------- MODE TIME ------------------------------
        elif self.method == "time":
            # Tambahkan sedikit variasi random (simulasi kecepatan berubah-ubah)
            return base_reward + np.random.randn()

        # -------------------------- MODE TRAFFIC BOX -----------------------
        elif self.method == "traffic_box":
            # Koordinat titik asal dan tujuan
            xs, ys = self.x[state], self.y[state]
            xe, ye = self.x[new_state], self.y[new_state]

            # Cek apakah garis perjalanan ini melewati zona macet
            intersections = self._calculate_box_intersection(xs, xe, ys, ye, self.box)

            if len(intersections) > 0:
                # Jika ada potongan garis di dalam kotak macet
                i1, i2 = intersections

                # Hitung panjang jalur yang dilalui di dalam zona macet
                distance_traffic = np.sqrt((i2[1] - i1[1]) ** 2 + (i2[0] - i1[0]) ** 2)

                # Tambahkan penalti berdasarkan panjang jalur dalam kotak dan tingkat kemacetan
                additional_reward = distance_traffic * self.traffic_intensity * np.random.rand()
            else:
                # Jika tidak melewati zona macet, tambahkan penalti kecil random
                additional_reward = np.random.rand()

            # Total reward = jarak dasar + penalti macet
            return base_reward + additional_reward

    # -------------------------------------------------------------------------
    # Fungsi bantu untuk cek apakah titik (x,y) berada di dalam kotak macet
    # -------------------------------------------------------------------------
    def _is_in_box(self, x, y, box):
        x_left, x_right, y_bottom, y_top = box
        return x >= x_left and x <= x_right and y >= y_bottom and y <= y_top

    # -------------------------------------------------------------------------
    # Hitung titik potong antara garis perjalanan dan kotak macet
    # -------------------------------------------------------------------------
    def _calculate_box_intersection(self, x1, x2, y1, y2, box):
        x_left, x_right, y_bottom, y_top = box   # Ambil batas kotak

        intersections = []                       # Tempat menyimpan titik potong (jika ada)

        # Cek potongan dengan sisi atas kotak
        i_top = self._calculate_point(x1, x2, y1, y2, y=y_top)
        if i_top > x_left and i_top < x_right:
            intersections.append((i_top, y_top))

        # Cek potongan dengan sisi bawah kotak
        i_bottom = self._calculate_point(x1, x2, y1, y2, y=y_bottom)
        if i_bottom > x_left and i_bottom < x_right:
            intersections.append((i_bottom, y_bottom))

        # Cek potongan dengan sisi kiri kotak
        i_left = self._calculate_point(x1, x2, y1, y2, x=x_left)
        if i_left > y_bottom and i_left < y_top:
            intersections.append((x_left, i_left))

        # Cek potongan dengan sisi kanan kotak
        i_right = self._calculate_point(x1, x2, y1, y2, x=x_right)
        if i_right > y_bottom and i_right < y_top:
            intersections.append((x_right, i_right))

        return intersections                      # Kembalikan daftar titik potong (bisa kosong)



# ============================================================
#  FUNGSI UNTUK MENJALANKAN SATU EPISODE PELATIHAN
# ============================================================
def run_episode(env, agent, verbose=1):

    s = env.reset()               # Reset environment → mulai dari state awal acak
    agent.reset_memory()          # Hapus riwayat titik yang pernah dikunjungi

    max_step = env.n_stops        # Batas langkah = jumlah titik pengiriman
    episode_reward = 0            # Total reward yang diperoleh selama episode
    i = 0                         # Counter langkah

    # Loop selama episode berjalan
    while i < max_step:

        #  Simpan state yang sedang dikunjungi (untuk menghindari pengulangan)
        agent.remember_state(s)

        #  Pilih aksi (yaitu titik berikutnya yang akan dikunjungi)
        a = agent.act(s)  # Menggunakan policy dari Q-table (epsilon-greedy)

        #  Lakukan aksi di environment → dapatkan state baru (s_next) dan reward (r)
        s_next, r, done = env.step(a)

        #  Tweak reward:
        # Karena reward asli adalah jarak, maka dikalikan -1
        # → jarak pendek = reward besar (lebih menguntungkan)
        # → jarak jauh = reward kecil (tidak efisien)
        r = -1 * r

        if verbose: 
            print(s_next, r, done)  # Tampilkan log jika verbose aktif

        #  Update Q-table dengan pengalaman baru (belajar dari interaksi)
        agent.train(s, a, r, s_next)

        #  Update variabel internal
        episode_reward += r         # Tambahkan reward ke total episode
        s = s_next                  # Pindah ke state berikutnya
        i += 1                      # Tambahkan langkah

        #  Hentikan episode jika semua titik sudah dikunjungi
        if done:
            break

    # Kembalikan hasil episode
    return env, agent, episode_reward






# ============================================================
#  AGENT KHUSUS UNTUK DELIVERY OPTIMIZATION
# ============================================================
class DeliveryQAgent(QAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   # Panggil konstruktor QAgent
        self.reset_memory()                 # Reset daftar state yang dikunjungi

    # ============================================================
    #  Fungsi untuk memilih aksi berikutnya (next stop)
    # ============================================================
    def act(self, s):
        # Ambil salinan Q-value untuk state saat ini
        q = np.copy(self.Q[s, :])

        # Hindari memilih titik yang sudah pernah dikunjungi
        # dengan memberi nilai -inf agar tidak terpilih lagi
        q[self.states_memory] = -np.inf

        # Gunakan strategi epsilon-greedy:
        #   - dengan probabilitas (1 - epsilon): pilih aksi terbaik (eksploitasi)
        #   - dengan probabilitas (epsilon): pilih aksi acak (eksplorasi)
        if np.random.rand() > self.epsilon:
            a = np.argmax(q)   # Pilih aksi dengan nilai Q tertinggi
        else:
            # Pilih acak dari titik-titik yang belum dikunjungi
            a = np.random.choice(
                [x for x in range(self.actions_size) if x not in self.states_memory]
            )

        return a   # Kembalikan aksi yang dipilih

    # ============================================================
    #  Simpan state yang sedang dikunjungi ke dalam memory
    # ============================================================
    def remember_state(self, s):
        self.states_memory.append(s)

    # ============================================================
    #  Reset memory (biasanya dipanggil saat mulai episode baru)
    # ============================================================
    def reset_memory(self):
        self.states_memory = []



# ============================================================
# 🏋️ FUNGSI UNTUK MENJALANKAN N EPISODE (PELATIHAN)
# ============================================================
def run_n_episodes(env, agent, name="training.gif", n_episodes=1000, render_each=10, fps=10):

    rewards = []   # Simpan total reward dari setiap episode
    imgs = []      # Simpan gambar environment (untuk membuat animasi GIF)

    #  Loop latihan sebanyak n_episodes
    for i in tqdm_notebook(range(n_episodes)):

        # Jalankan satu episode (agent berinteraksi dengan environment)
        env, agent, episode_reward = run_episode(env, agent, verbose=0)

        # Simpan total reward dari episode ini
        rewards.append(episode_reward)

        # Setiap beberapa episode, render environment jadi gambar
        if i % render_each == 0:
            img = env.render(return_img=True)
            imgs.append(img)

    # ============================================================
    #  Tampilkan grafik hasil pelatihan (reward per episode)
    # ============================================================
    plt.figure(figsize=(15, 3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.show()

    # ============================================================
    #  Simpan semua gambar jadi animasi GIF
    # ============================================================
    imageio.mimsave(name, imgs, fps=fps)

    # Kembalikan environment dan agent yang sudah dilatih
    return env, agent
