# =========================================================================
# ==                          IMPOR PUSTAKA                               ==
# =========================================================================
# Impor pustaka-pustaka dasar untuk analisis data, matematika, dan visualisasi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook # Untuk membuat progress bar di notebook
from scipy.spatial.distance import cdist # Untuk menghitung jarak Euclidean antar titik
import imageio # Untuk membuat animasi GIF dari gambar
from matplotlib.patches import Rectangle # Untuk menggambar persegi (zona macet)
from matplotlib.collections import PatchCollection # Untuk mengelola koleksi bentuk

# =========================================================================
# ==                      KONFIGURASI VISUALISASI                         ==
# =========================================================================
# Mengatur gaya visual default untuk semua plot matplotlib yang akan dibuat
plt.style.use("seaborn-v0_8-darkgrid")

import sys
# Menambahkan direktori induk ke path agar bisa mengimpor modul dari sana
sys.path.append("../") 
# Impor kelas QAgent dasar dari pustaka RL yang sudah ada
from rl.agents.q_agent import QAgent


# =========================================================================
# ==                 CLASS: DeliveryEnvironment (Dunia Simulasi)          ==
# =========================================================================
# Kelas ini merepresentasikan "dunia" atau "peta kota" tempat agen beroperasi.
# Ia bertanggung jawab untuk menciptakan titik-titik, menghitung jarak,
# dan memberikan reward kepada agen.
class DeliveryEnvironment(object):
    # --- Metode Inisialisasi (dipanggil saat objek dibuat) ---
    def __init__(self,n_stops = 10,max_box = 10,method = "distance",**kwargs):

        print(f"Initialized Delivery Environment with {n_stops} random stops")
        print(f"Target metric for optimization is {method}")

        # Inisialisasi atribut-atribut dasar dari environment
        self.n_stops = n_stops                  # Jumlah total titik yang harus dikunjungi
        self.action_space = self.n_stops        # Jumlah aksi yang mungkin (memilih salah satu dari n_stops)
        self.observation_space = self.n_stops   # Jumlah state yang mungkin (berada di salah satu dari n_stops)
        self.max_box = max_box                  # Ukuran area simulasi (misal: 10x10)
        self.stops = []                         # Daftar untuk menyimpan urutan titik yang sudah dikunjungi agen
        self.method = method                    # Metode perhitungan reward ('distance', 'time', 'traffic_box')

        # Memulai proses pembuatan dunia simulasi
        self._generate_constraints(**kwargs)    # Buat zona macet (jika metodenya 'traffic_box')
        self._generate_stops()                  # Buat koordinat acak untuk semua titik pemberhentian
        self._generate_q_values()               # Hitung matriks jarak/waktu antar semua titik
        self.render()                           # Gambar kondisi awal peta (sebelum agen bergerak)

        # Siapkan untuk episode pertama
        self.reset()                            # Reset simulasi dan letakkan agen di titik awal acak


    # --- Metode untuk membuat zona macet (opsional) ---
    def _generate_constraints(self,box_size = 0.2,traffic_intensity = 5):
        # Hanya berjalan jika metodenya adalah 'traffic_box'
        if self.method == "traffic_box":
            # Tentukan posisi kiri bawah dari kotak macet secara acak
            x_left = np.random.rand() * (self.max_box) * (1-box_size)
            y_bottom = np.random.rand() * (self.max_box) * (1-box_size)

            # Tentukan posisi kanan atas berdasarkan ukuran kotak
            x_right = x_left + np.random.rand() * box_size * self.max_box
            y_top = y_bottom + np.random.rand() * box_size * self.max_box

            # Simpan koordinat kotak [kiri, kanan, bawah, atas] dan tingkat kemacetan
            self.box = (x_left,x_right,y_bottom,y_top)
            self.traffic_intensity = traffic_intensity 


    # --- Metode untuk membuat koordinat titik-titik pemberhentian ---
    def _generate_stops(self):
        # Jika metodenya 'traffic_box', pastikan titik tidak berada di dalam zona macet
        if self.method == "traffic_box":
            points = []
            while len(points) < self.n_stops:
                x,y = np.random.rand(2)*self.max_box
                # Cek apakah titik yang baru dibuat ada di dalam kotak macet
                if not self._is_in_box(x,y,self.box):
                    points.append((x,y))
            xy = np.array(points)
        else:
            # Jika tidak ada zona macet, buat saja koordinat acak di seluruh area
            xy = np.random.rand(self.n_stops,2)*self.max_box

        # Simpan koordinat x dan y ke dalam atribut terpisah
        self.x = xy[:,0]
        self.y = xy[:,1]


    # --- Metode untuk menghitung matriks "biaya" antar titik ---
    def _generate_q_values(self,box_size = 0.2):
        # Metode ini menghitung "biaya" dasar untuk berpindah dari setiap titik ke setiap titik lainnya.
        # Biaya ini bisa berupa jarak, waktu, dll.

        # Jika metodenya berbasis jarak
        if self.method in ["distance","traffic_box"]:
            xy = np.column_stack([self.x,self.y]) # Gabungkan kembali array x dan y
            # cdist(xy, xy) akan membuat matriks di mana elemen (i, j) adalah jarak dari titik i ke titik j
            self.q_stops = cdist(xy,xy)
        # Jika metodenya berbasis waktu
        elif self.method=="time":
            # Buat matriks acak untuk mensimulasikan waktu tempuh yang bervariasi
            self.q_stops = np.random.rand(self.n_stops,self.n_stops)*self.max_box
            # Waktu tempuh dari sebuah titik ke dirinya sendiri adalah 0
            np.fill_diagonal(self.q_stops,0)
        else:
            raise Exception("Method not recognized")
    

    # --- Metode untuk memvisualisasikan kondisi environment saat ini ---
    def render(self, return_img=False):
        # Membuat area plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("Delivery Stops", fontsize=14, fontweight='bold')
        ax.set_facecolor("#f9f9f9")

        # Plot semua titik pemberhentian sebagai titik-titik merah
        ax.scatter(self.x, self.y, c="#e74c3c", s=60, label="Stops")

        # Jika episode sudah dimulai (ada titik di self.stops)
        if self.stops:
            # Dapatkan koordinat titik awal
            sx, sy = self._get_xy(initial=True)
            # Beri label "START" di dekat titik awal
            ax.annotate("START", xy=(sx, sy), xytext=(sx + 0.1, sy - 0.05),
                    fontweight="bold", color="#2c3e50")

        # Jika agen sudah mengunjungi lebih dari satu titik
        if len(self.stops) > 1:
            # Gambar garis putus-putus biru yang menghubungkan titik-titik yang sudah dikunjungi
            ax.plot(self.x[self.stops], self.y[self.stops],
                c="#2980b9", linewidth=2, linestyle="--")
            # Dapatkan koordinat titik terakhir yang dikunjungi
            ex, ey = self._get_xy(initial=False)
            # Beri label "END" di dekat titik tersebut
            ax.annotate("END", xy=(ex, ey), xytext=(ex + 0.1, ey - 0.05),
                    fontweight="bold", color="#2c3e50")

        # Jika ada zona macet, gambar sebagai persegi transparan
        if hasattr(self, "box"):
            left, bottom = self.box[0], self.box[2]
            width, height = self.box[1] - self.box[0], self.box[3] - self.box[2]
            rect = Rectangle((left, bottom), width, height, facecolor="#e74c3c", alpha=0.2)
            ax.add_patch(rect)

        # Bersihkan sumbu X dan Y dan tampilkan legenda
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()

        # Jika diminta untuk mengembalikan sebagai gambar (untuk membuat GIF)
        if return_img:
            fig.canvas.draw()
            # Konversi kanvas plot menjadi array numpy (gambar)
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig) # Tutup plot agar tidak ditampilkan langsung
            return image
        else:
            # Jika tidak, tampilkan plotnya langsung
            plt.show()


    # --- Metode untuk mereset simulasi ke kondisi awal ---
    def reset(self):
        # Kosongkan daftar titik yang sudah dikunjungi
        self.stops = []

        # Pilih titik awal baru secara acak
        first_stop = np.random.randint(self.n_stops)
        self.stops.append(first_stop)

        # Kembalikan titik awal sebagai state pertama
        return first_stop


    # --- Metode untuk menjalankan satu langkah (step) dalam simulasi ---
    def step(self,destination):
        # Dapatkan state saat ini (titik terakhir yang dikunjungi)
        state = self._get_state()
        new_state = destination # State baru adalah tujuan yang dipilih agen

        # Hitung reward (biaya) untuk perjalanan dari 'state' ke 'new_state'
        reward = self._get_reward(state,new_state)

        # Tambahkan tujuan baru ke dalam daftar perjalanan
        self.stops.append(destination)
        # Cek apakah episode selesai (semua titik sudah dikunjungi)
        done = len(self.stops) == self.n_stops

        # Kembalikan hasil dari langkah ini
        return new_state,reward,done
    

    # --- Metode bantu untuk mendapatkan state saat ini ---
    def _get_state(self):
        # State adalah elemen terakhir dalam daftar 'stops'
        return self.stops[-1]


    # --- Metode bantu untuk mendapatkan koordinat (x,y) dari sebuah titik ---
    def _get_xy(self,initial = False):
        # Jika 'initial' adalah True, ambil titik pertama. Jika tidak, ambil titik terakhir.
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x,y


    # --- Metode untuk menghitung reward berdasarkan metode yang dipilih ---
    def _get_reward(self,state,new_state):
        # Ambil "biaya" dasar dari matriks yang sudah dihitung sebelumnya
        base_reward = self.q_stops[state,new_state]

        if self.method == "distance":
            # Jika metodenya jarak, reward adalah jarak itu sendiri
            return base_reward
        elif self.method == "time":
            # Jika metodenya waktu, tambahkan sedikit noise acak untuk simulasi
            return base_reward + np.random.randn()
        elif self.method == "traffic_box":
            # Jika ada zona macet, hitung penalti tambahan
            xs,ys = self.x[state],self.y[state]
            xe,ye = self.x[new_state],self.y[new_state]
            # Cek apakah garis perjalanan memotong kotak macet
            intersections = self._calculate_box_intersection(xs,xe,ys,ye,self.box)
            if len(intersections) > 0:
                # Jika memotong, hitung jarak di dalam zona macet
                i1,i2 = intersections
                distance_traffic = np.sqrt((i2[1]-i1[1])**2 + (i2[0]-i1[0])**2)
                # Hitung penalti macet berdasarkan jarak dan tingkat kemacetan
                additional_reward = distance_traffic * self.traffic_intensity * np.random.rand()
            else:
                # Jika tidak, beri penalti acak yang kecil
                additional_reward = np.random.rand()

            # Reward total adalah biaya dasar ditambah penalti macet
            return base_reward + additional_reward


    # --- Metode statis (tidak butuh 'self') untuk matematika garis ---
    @staticmethod
    def _calculate_point(x1,x2,y1,y2,x = None,y = None):
        # Fungsi ini menghitung titik potong sebuah garis dengan garis horizontal atau vertikal.
        if y1 == y2: # Garis horizontal
            return y1
        elif x1 == x2: # Garis vertikal
            return x1
        else:
            # Hitung gradien (a) dan intercept (b) dari persamaan garis y = ax + b
            a = (y2-y1)/(x2-x1)
            b = y2 - a * x2

            if x is None: # Jika diberi y, hitung x
                x = (y-b)/a
                return x
            elif y is None: # Jika diberi x, hitung y
                y = a*x+b
                return y
            else:
                raise Exception("Provide x or y")


    # --- Metode bantu untuk mengecek apakah titik ada di dalam kotak ---
    def _is_in_box(self,x,y,box):
        x_left,x_right,y_bottom,y_top = box
        # Cek apakah koordinat x dan y berada di dalam batas-batas kotak
        return x >= x_left and x <= x_right and y >= y_bottom and y <= y_top


    # --- Metode bantu untuk menemukan titik potong garis dengan kotak ---
    def _calculate_box_intersection(self,x1,x2,y1,y2,box):
        x_left,x_right,y_bottom,y_top = box
        intersections = []

        # Hitung titik potong dengan setiap sisi kotak
        # Sisi atas
        i_top = self._calculate_point(x1,x2,y1,y2,y=y_top)
        if i_top > x_left and i_top < x_right:
            intersections.append((i_top,y_top))
        # Sisi bawah
        i_bottom = self._calculate_point(x1,x2,y1,y2,y=y_bottom)
        if i_bottom > x_left and i_bottom < x_right:
            intersections.append((i_bottom,y_bottom))
        # Sisi kiri
        i_left = self._calculate_point(x1,x2,y1,y2,x=x_left)
        if i_left > y_bottom and i_left < y_top:
            intersections.append((x_left,i_left))
        # Sisi kanan
        i_right = self._calculate_point(x1,x2,y1,y2,x=x_right)
        if i_right > y_bottom and i_right < y_top:
            intersections.append((x_right,i_right))

        return intersections


# =========================================================================
# ==                 FUNGSI: run_episode (Menjalankan 1 Episode)          ==
# =========================================================================
# Fungsi ini mengelola satu siklus lengkap perjalanan, dari titik awal
# sampai semua titik terkunjungi. Di sinilah proses belajar terjadi.
def run_episode(env,agent,verbose = 1):

    # Mulai episode baru: reset environment dan dapatkan state awal
    s = env.reset()
    # Kosongkan ingatan jangka pendek agen
    agent.reset_memory()

    max_step = env.n_stops  # Jumlah maksimum langkah adalah jumlah titik
    episode_reward = 0      # Inisialisasi total reward untuk episode ini
    
    i = 0
    # Loop sampai semua titik dikunjungi
    while i < max_step:

        # Ingat state saat ini (untuk menghindari kunjungan berulang)
        agent.remember_state(s)

        # Agen memilih aksi (tujuan berikutnya) berdasarkan kebijakannya
        a = agent.act(s)
        
        # Jalankan aksi di environment dan dapatkan hasilnya
        s_next,r,done = env.step(a)

        # "Membalik" reward. Jarak adalah "biaya", jadi kita ingin meminimalkannya.
        # Dalam RL, agen mencoba memaksimalkan reward.
        # Jadi, kita ubah jarak menjadi negatif. Jarak kecil -> reward besar (misal -5 > -10).
        r = -1 * r
        
        if verbose: print(s_next,r,done)
        
        # Momen paling penting: Agen BELAJAR dari pengalamannya.
        # Ia memperbarui Q-Table berdasarkan (state, aksi, reward, state_berikutnya).
        agent.train(s,a,r,s_next)
        
        # Pindah ke state berikutnya dan akumulasi reward
        episode_reward += r
        s = s_next
        
        i += 1
        # Jika environment memberi sinyal 'done', episode selesai
        if done:
            break
            
    return env,agent,episode_reward


# =========================================================================
# ==      CLASS: DeliveryQAgent (Agen Spesialis TSP)                      ==
# =========================================================================
# Ini adalah turunan dari QAgent dasar, tetapi dengan modifikasi khusus
# untuk menyelesaikan TSP: yaitu, tidak mengunjungi titik yang sama dua kali.
class DeliveryQAgent(QAgent):

    # --- Metode Inisialisasi ---
    def __init__(self,*args,**kwargs):
        # Panggil inisialisasi dari kelas induknya (QAgent)
        super().__init__(*args,**kwargs)
        # Siapkan ingatan jangka pendek
        self.reset_memory()

    # --- Metode untuk memilih aksi (ditimpa dari QAgent) ---
    def act(self,s):

        # Salin baris Q-value untuk state saat ini dari Q-Table
        q = np.copy(self.Q[s,:])

        # ### TRIK UTAMA UNTUK TSP ###
        # Set nilai Q untuk semua titik yang sudah dikunjungi menjadi negatif tak hingga (-inf).
        # Ini memastikan bahwa agen TIDAK AKAN PERNAH memilih titik-titik ini lagi.
        q[self.states_memory] = -np.inf

        # Gunakan kebijakan epsilon-greedy
        if np.random.rand() > self.epsilon:
            # Eksploitasi: Pilih aksi (titik) dengan nilai Q tertinggi dari yang tersisa
            a = np.argmax(q)
        else:
            # Eksplorasi: Pilih aksi secara acak dari daftar titik yang BELUM dikunjungi
            a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_memory])

        return a


    # --- Metode untuk mengingat state yang baru dikunjungi ---
    def remember_state(self,s):
        self.states_memory.append(s)

    # --- Metode untuk mengosongkan ingatan di awal episode baru ---
    def reset_memory(self):
        self.states_memory = []


# =========================================================================
# ==      FUNGSI: run_n_episodes (Menjalankan Seluruh Proses Pelatihan)     ==
# =========================================================================
# Fungsi ini adalah 'pelatih' utama. Ia akan memanggil run_episode
# berulang kali untuk melatih agen.
def run_n_episodes(env,agent,name="training.gif",n_episodes=1000,render_each=10,fps=10):

    # Siapkan daftar untuk menyimpan hasil
    rewards = []
    imgs = []

    # Loop pelatihan utama
    for i in tqdm_notebook(range(n_episodes)):

        # Jalankan satu episode penuh
        env,agent,episode_reward = run_episode(env,agent,verbose = 0)
        # Simpan total reward (jarak negatif) dari episode tersebut
        rewards.append(episode_reward)
        
        # Setiap 'render_each' episode, ambil gambar kondisi environment
        if i % render_each == 0:
            img = env.render(return_img = True)
            imgs.append(img)

    # Setelah pelatihan selesai, tampilkan grafik performa
    plt.figure(figsize = (15,3))
    plt.title("Rewards over training")
    # Plot total reward per episode. Jika agen belajar, grafik ini akan NAIK
    # (karena reward adalah jarak negatif, jadi -50 lebih baik dari -100).
    plt.plot(rewards)
    plt.show()

    # Gabungkan semua gambar yang disimpan menjadi satu animasi GIF
    imageio.mimsave(name,imgs,fps = fps)

    # Kembalikan environment dan agen yang sudah terlatih
    return env,agent