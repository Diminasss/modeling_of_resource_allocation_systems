"""
Лабораторная работа №2
Анализ системы множественного доступа с топологией типа звезда
Вариант 2: Окумура-Хата (large city), R=3000 м, PTX=160 Вт, f0=1800 МГц, kn=2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─── Параметры варианта 2 ───────────────────────────────────────────────────
R        = 3000           # Радиус соты, м
PTX_W    = 160            # Мощность передатчика БС, Вт
f0       = 1800           # Несущая частота, МГц
kn       = 2              # Коэффициент шума приёмника (линейный)

H_TE     = 30             # Высота антенны БС, м
H_RE     = 1.5            # Высота антенны АБ, м
SIGMA    = 8.0            # СКО лог-нормального затенения, дБ
DELTA_F  = 200e3          # Полоса пропускания на одного АБ, Гц (200 кГц)
T_SLOT   = 1e-3           # Длительность слота, с (1 мс)
K_SLOTS  = 1000           # Количество временных слотов
K_B      = 1.38e-23       # Постоянная Больцмана, Дж/К
T0       = 290            # Температура, К
PKT_SIZE = 1024           # Размер пакета, байт (1 кбайт)

N_LIST   = [8, 16, 64, 128]          # Числа абонентов для моделирования
P_VALS   = np.arange(0.1, 1.0, 0.1) # Значения параметра p

# Производные константы
P_NOISE_W = kn * K_B * T0 * DELTA_F
PTX_DBM   = 10 * np.log10(PTX_W * 1e3)
COLORS    = plt.cm.tab10.colors      # берём столько цветов, сколько нужно


# ═══════════════════════════════════════════════════════════════════════════
#  Модель канала
# ═══════════════════════════════════════════════════════════════════════════

def okumura_hata_large_city(d_m):
    """Среднее затухание (дБ) по модели Окумура-Хата, large city."""
    d_km = np.maximum(d_m, 10) / 1000.0
    a_hre = 3.2 * (np.log10(11.75 * H_RE)) ** 2 - 4.97
    return (69.55
            + 26.16 * np.log10(f0)
            - 13.82 * np.log10(H_TE)
            - a_hre
            + (44.9 - 6.55 * np.log10(H_TE)) * np.log10(d_km))


def place_subscribers(N):
    """Равномерное случайное размещение N АБ внутри круга радиуса R."""
    r = R * np.sqrt(np.random.uniform(0, 1, N))
    return np.maximum(r, 10.0)


def channel_snr(L_mean, K):
    """
    Генерация SNR для N абонентов и K слотов.
    Возвращает матрицу SNR (N, K) в линейном масштабе.
    """
    N = len(L_mean)
    xi    = np.random.normal(0, SIGMA, size=(N, K))
    L_ik  = L_mean[:, np.newaxis] + xi
    P_rx  = PTX_DBM - L_ik
    P_rx_W = 10 ** ((P_rx - 30) / 10)
    return P_rx_W / P_NOISE_W


def slot_volume(snr):
    """Объём данных за слот (байт) по формуле Шеннона. snr — матрица (N, K)."""
    rate = DELTA_F * np.log2(1 + snr)
    return rate * T_SLOT / 8


# ═══════════════════════════════════════════════════════════════════════════
#  Алгоритмы планирования
# ═══════════════════════════════════════════════════════════════════════════

def simulate_round_robin(N, p):
    """
    Round-Robin: в слоте k обслуживается абонент k mod N.
    Возвращает среднее суммарное значение буфера по K слотам (байт).
    """
    d      = place_subscribers(N)
    L_mean = okumura_hata_large_city(d)
    V_ik   = slot_volume(channel_snr(L_mean, K_SLOTS))
    A_ik   = np.random.geometric(p, size=(N, K_SLOTS)) * PKT_SIZE

    buf   = np.zeros(N)
    total = 0.0
    for k in range(K_SLOTS):
        buf += A_ik[:, k]
        i      = k % N
        buf[i] = max(0.0, buf[i] - min(buf[i], V_ik[i, k]))
        total += buf.sum()
    return total / K_SLOTS


def simulate_proportional_fair(N, p, window=20):
    """
    Proportional Fair: абонент выбирается по max(V_ik / avg_C_i).
    Возвращает среднее суммарное значение буфера по K слотам (байт).
    """
    d      = place_subscribers(N)
    L_mean = okumura_hata_large_city(d)
    V_ik   = slot_volume(channel_snr(L_mean, K_SLOTS))
    A_ik   = np.random.geometric(p, size=(N, K_SLOTS)) * PKT_SIZE

    buf      = np.zeros(N)
    avg_rate = np.full(N, np.mean(V_ik))
    alpha    = 1 / window
    total    = 0.0
    for k in range(K_SLOTS):
        buf += A_ik[:, k]
        priorities   = V_ik[:, k] / (avg_rate + 1e-9)
        i            = int(np.argmax(priorities))
        avg_rate     = (1 - alpha) * avg_rate
        avg_rate[i] += alpha * V_ik[i, k]
        buf[i]       = max(0.0, buf[i] - min(buf[i], V_ik[i, k]))
        total       += buf.sum()
    return total / K_SLOTS


# ═══════════════════════════════════════════════════════════════════════════
#  Сбор результатов
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation():
    """
    Прогоняет Round-Robin и Proportional Fair для всех N из N_LIST
    и всех p из P_VALS.
    Возвращает два словаря: rr_results[N] и pf_results[N] — массивы по p.
    """
    rr_results, pf_results = {}, {}
    for N in N_LIST:
        rr_buf, pf_buf = [], []
        for p in P_VALS:
            rr = simulate_round_robin(N, p)
            pf = simulate_proportional_fair(N, p)
            rr_buf.append(rr)
            pf_buf.append(pf)
            print(f"  N={N:4d}, p={p:.1f} | RR: {rr/1024:8.1f} кБ | PF: {pf/1024:8.1f} кБ")
        rr_results[N] = np.array(rr_buf)
        pf_results[N] = np.array(pf_buf)
    return rr_results, pf_results


# ═══════════════════════════════════════════════════════════════════════════
#  Построение графиков
# ═══════════════════════════════════════════════════════════════════════════

def _draw_subplot(ax, results, title):
    """Рисует один подграфик для переданного словаря результатов."""
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    for idx, N in enumerate(N_LIST):
        ax.plot(
            P_VALS,
            results[N] / 1024,
            color=COLORS[idx % len(COLORS)],
            marker=markers[idx % len(markers)],
            label=f"N = {N}",
        )
    ax.set_xlabel("Параметр p", fontsize=12)
    ax.set_ylabel("Средний суммарный объём данных в буфере, кБ", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(P_VALS)


def plot_results(rr_results, pf_results, save_path=None):
    """Строит два подграфика: Round-Robin и Proportional Fair."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Лаб. работа №2. Модель Окумура-Хата (large city)\n"
        f"R={R} м, PTX={PTX_W} Вт, f0={f0} МГц, kn={kn}",
        fontsize=13,
    )
    _draw_subplot(axes[0], rr_results, "Алгоритм Round-Robin")
    _draw_subplot(axes[1], pf_results, "Алгоритм Proportional Fair")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Графики сохранены: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Точка входа
# ═══════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    print("=== Моделирование ===")
    rr_results, pf_results = run_simulation()
    plot_results(rr_results, pf_results, save_path="lab2_graphs.png")


if __name__ == "__main__":
    main()