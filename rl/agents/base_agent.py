#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
--------------------------------------------------------------------
 BASE AGENT (KELAS DASAR UNTUK SEMUA AGEN RL)

 Tujuan:
    - Menjadi template (kerangka dasar) untuk agent reinforcement learning.
    - Belum memiliki kemampuan belajar atau mengambil keputusan.
    - Hanya menyediakan fungsi umum yang sering digunakan oleh turunan lain
      seperti QAgent dan DeliveryQAgent.

 Peran dalam sistem:
    - QAgent akan mewarisi (inherit) class ini dan menambahkan logika pembelajaran.
    - DeliveryQAgent juga akan mewarisi QAgent dan menambahkan aturan tambahan
      (misalnya tidak boleh mengunjungi titik yang sama dua kali).

--------------------------------------------------------------------
"""

import numpy as np

class Agent(object):
    def __init__(self):
        # Konstruktor kosong (belum ada variabel khusus di kelas dasar)
        pass

    # ============================================================
    #  Fungsi: expand_state_vector
    # ============================================================
    def expand_state_vector(self, state):
        """
        Fungsi ini digunakan untuk memastikan bahwa input state
        memiliki dimensi (shape) yang benar sebelum diproses oleh model.
        
        Jika state berbentuk 1D (misalnya [3, 5]) atau 3D (misalnya gambar),
        maka fungsi ini akan menambahkan satu dimensi ekstra di depan
        agar sesuai dengan format input model (misalnya [[3, 5]]).
        """
        if len(state.shape) == 1 or len(state.shape) == 3:
            return np.expand_dims(state, axis=0)  # Tambahkan dimensi di depan
        else:
            return state  # Jika sudah dalam format benar, biarkan

    # ============================================================
    #  Fungsi: remember
    # ============================================================
    def remember(self, *args):
        """
        Menyimpan pengalaman ke dalam memory agent.
        Biasanya pengalaman berisi tuple:
        (state, action, reward, next_state)
        
        Fungsi ini akan dipanggil setiap kali agent
        berinteraksi dengan environment.
        """
        self.memory.save(args)
