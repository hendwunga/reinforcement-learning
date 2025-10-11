#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
--------------------------------------------------------------------
⚙️ Q-LEARNING AGENT MODULE
--------------------------------------------------------------------
📘 Tujuan:
    - Mewakili agen (AI) yang belajar mengambil keputusan menggunakan metode Q-Learning.
    - Agen belajar nilai (Q-value) dari setiap kombinasi state–action.
    - Semakin sering latihan, semakin baik agen dalam memilih aksi optimal.

🧩 Konsep penting:
    - state  = kondisi lingkungan (misalnya titik lokasi kurir)
    - action = keputusan yang diambil (misalnya titik berikutnya)
    - reward = umpan balik dari environment (misalnya waktu tempuh atau jarak)
    - Q-table = tabel yang menyimpan nilai seberapa baik suatu aksi diambil pada suatu state
--------------------------------------------------------------------
👨‍💻 Author: theo.alves.da.costa@gmail.com
🔗 Repo: https://github.com/theolvs
Started: 25/08/2017
--------------------------------------------------------------------
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time

from rl import utils
from rl.memory import Memory
from rl.agents.base_agent import Agent


# ============================================================
# 🤖 KELAS: QAgent
# ============================================================
# Kelas ini merupakan implementasi agen pembelajaran berbasis Q-Learning.
# Agen ini memperluas (inherit) kelas dasar Agent dan menambahkan:
#   - Q-table untuk menyimpan nilai tiap (state, action)
#   - Parameter pembelajaran (epsilon, gamma, lr)
#   - Fungsi act() untuk memilih aksi (exploration vs exploitation)
#   - Fungsi train() untuk memperbarui Q-table berdasarkan reward
# ============================================================

class QAgent(Agent):
    def __init__(self,
                 states_size,       # Jumlah total state (keadaan yang mungkin)
                 actions_size,      # Jumlah total action (aksi yang mungkin)
                 epsilon=1.0,       # Probabilitas eksplorasi (awal: 100%)
                 epsilon_min=0.01,  # Batas minimum eksplorasi
                 epsilon_decay=0.999,# Laju penurunan eksplorasi per episode
                 gamma=0.95,        # Faktor diskonto (pentingnya reward masa depan)
                 lr=0.8):           # Learning rate (kecepatan belajar)
        
        #  Simpan semua parameter ke dalam atribut objek
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr

        #  Inisialisasi Q-table berukuran [state, action]
        self.Q = self.build_model(states_size, actions_size)

    # ============================================================
    #  MEMBANGUN MODEL Q-TABLE
    # ============================================================
    def build_model(self, states_size, actions_size):
        # Q-table diisi dengan nol awalnya
        Q = np.zeros([states_size, actions_size])
        return Q

    # ============================================================
    #  LATIH AGENT (UPDATE Q-TABLE)
    # ============================================================
    def train(self, s, a, r, s_next):
        """
        Update nilai Q berdasarkan formula Q-Learning:
        Q(s,a) = Q(s,a) + lr * [r + gamma * max(Q(s_next,:)) - Q(s,a)]
        """
        self.Q[s, a] = self.Q[s, a] + self.lr * (
            r + self.gamma * np.max(self.Q[s_next, :]) - self.Q[s, a]
        )

        # Kurangi nilai epsilon (agen makin lama makin sedikit eksplorasi)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ============================================================
    #  PILIH AKSI BERDASARKAN POLICY (EPSILON-GREEDY)
    # ============================================================
    def act(self, s):
        q = self.Q[s, :]  # Ambil semua nilai Q dari state saat ini

        # Dengan peluang (1 - epsilon), pilih aksi terbaik (eksploitasi)
        if np.random.rand() > self.epsilon:
            a = np.argmax(q)  # Ambil indeks aksi dengan nilai Q tertinggi
        else:
            # Dengan peluang epsilon, pilih aksi acak (eksplorasi)
            a = np.random.randint(self.actions_size)

        return a  # Kembalikan aksi yang dipilih
