#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
--------------------------------------------------------------------
MEMORY (PENYIMPANAN PENGALAMAN RL)

Tujuan:
    - Menyimpan semua pengalaman agent dalam bentuk antrian (deque).
    - Setiap pengalaman biasanya berupa tuple (state, action, reward, next_state).
    - Digunakan agar agent bisa belajar dari pengalaman masa lalu.

Analogi:
    Seperti “ingatannya” agent. 
    Saat agent berjalan di environment, setiap langkah (pengalaman)
    disimpan di sini, lalu digunakan kembali saat update Q-Table.

--------------------------------------------------------------------
"""

from collections import deque

class Memory(object):
    def __init__(self, max_memory=2000):
        """
        Inisialisasi memori dengan kapasitas maksimum.
        Jika memori sudah penuh, data lama otomatis dihapus (FIFO).
        """
        self.cache = deque(maxlen=max_memory)  # deque = struktur antrian efisien

    # ============================================================
    # 🔹 Fungsi: save
    # ============================================================
    def save(self, args):
        """
        Simpan satu pengalaman ke dalam cache.
        Biasanya args = (state, action, reward, next_state)
        """
        self.cache.append(args)

    # ============================================================
    # 🔹 Fungsi: empty_cache
    # ============================================================
    def empty_cache(self):
        """
        Menghapus semua isi memori dengan cara menginisialisasi ulang.
        """
        self.__init__()
