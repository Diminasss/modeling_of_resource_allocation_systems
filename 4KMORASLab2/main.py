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
R       = 3000          # Радиус соты, м
PTX_W   = 160           # Мощность передатчика БС, Вт
f0      = 1800          # Несущая частота, МГц
kn      = 2             # Коэффициент шума приёмника (линейный)

# ─── Прочие константы ───────────────────────────────────────────────────────
h_te    = 30            # Высота антенны БС, м
h_re    = 1.5           # Высота антенны АБ, м
sigma   = 8.0           # СКО лог-нормального затенения, дБ
delta_f = 200e3         # Полоса пропускания на одного АБ, Гц (200 кГц)
T_slot  = 1e-3          # Длительность слота, с (1 мс)
K       = 1000          # Количество временных слотов
k_B     = 1.38e-23      # Постоянная Больцмана, Дж/К
T0      = 290           # Температура, К
N_list  = [8, 16, 64]   # Варианты числа абонентов
p_vals  = np.arange(0.1, 1.0, 0.1)   # Значения параметра p


# ═══════════════════════════════════════════════════════════════════════════
#  1. МОДЕЛЬ ОКУМУРА-ХАТА (LARGE CITY, f > 300 МГц)
# ═══════════════════════════════════════════════════════════════════════════
def okumura_hata_large_city(d_m, fc=f0, hte=h_te, hre=h_re):
    """
    Среднее затухание в канале по модели Окумура-Хата для большого города.
    d_m  - расстояние от БС до АБ, м (скалярное или массив)
    fc   - несущая частота, МГц
    hte  - высота антенны БС, м
    hre  - высота антенны АБ, м
    Возвращает: затухание L, дБ
    """
    d_km = d_m / 1000.0                 # перевод в км

    # Корректирующий множитель для большого города (fc >= 300 МГц)
    a_hre = 3.2 * (np.log10(11.75 * hre))**2 - 4.97   # дБ

    L = (69.55
         + 26.16 * np.log10(fc)
         - 13.82 * np.log10(hte)
         - a_hre
         + (44.9 - 6.55 * np.log10(hte)) * np.log10(d_km))
    return L


# ═══════════════════════════════════════════════════════════════════════════
#  2. РАЗМЕЩЕНИЕ АБОНЕНТОВ В КРУГЕ (метод равномерного распределения)
# ═══════════════════════════════════════════════════════════════════════════
def place_subscribers(N, R):
    """
    Случайное размещение N АБ внутри круга радиуса R.
    Возвращает расстояния d[i] от БС до каждого АБ (массив длиной N).
    """
    theta = np.random.uniform(0, 2 * np.pi, N)
    r     = R * np.sqrt(np.random.uniform(0, 1, N))   # равномерно по площади
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    d = np.sqrt(x**2 + y**2)
    # Минимальное расстояние 10 м, чтобы избежать log(0)
    d = np.maximum(d, 10.0)
    return d


# ═══════════════════════════════════════════════════════════════════════════
#  3. МОЩНОСТЬ ШУМА И ПРИНЯТАЯ МОЩНОСТЬ
# ═══════════════════════════════════════════════════════════════════════════
# Мощность теплового шума с учётом коэффициента шума
P_noise_W   = kn * k_B * T0 * delta_f   # Вт
P_noise_dBm = 10 * np.log10(P_noise_W * 1e3)   # дБм

# Мощность передатчика
PTX_dBm = 10 * np.log10(PTX_W * 1e3)   # дБм


def received_power_dBm(L_dB):
    """Мощность на входе АБ, дБм (скалярная или массив)."""
    return PTX_dBm - L_dB


def snr_linear(P_rx_dBm):
    """ОСШ в линейном масштабе."""
    P_rx_W    = 10**((P_rx_dBm - 30) / 10)
    return P_rx_W / P_noise_W


# ═══════════════════════════════════════════════════════════════════════════
#  4. МАКСИМАЛЬНАЯ СКОРОСТЬ (Формула Шеннона) и ОБЪЁМ ДАННЫХ ЗА СЛОТ
# ═══════════════════════════════════════════════════════════════════════════
def max_rate(snr):
    """Максимальная скорость передачи данных, бит/с."""
    return delta_f * np.log2(1 + snr)


def data_volume_slot(rate_bps):
    """Объём данных, передаваемых за один слот, байт."""
    return rate_bps * T_slot / 8   # бит/с × с / 8 = байт


# ═══════════════════════════════════════════════════════════════════════════
#  5. МОДЕЛИРОВАНИЕ БУФЕРА  (Round-Robin)
# ═══════════════════════════════════════════════════════════════════════════
PACKET_SIZE = 1024   # размер пакета, байт (1 кбайт)


def simulate_round_robin(N, p, K=K):
    """
    Моделирование передачи данных по Round-Robin.
    N  - число абонентов
    p  - параметр геометрического распределения прихода пакетов
    K  - число слотов
    Возвращает средний суммарный объём данных в буфере по слотам (байт).
    """
    # Размещение абонентов
    d = place_subscribers(N, R)

    # Среднее затухание канала для каждого АБ (дБ)
    L_mean = okumura_hata_large_city(d)

    # Генерация затенения: L_ik = L_i + xi_ik,  xi ~ N(0, sigma)
    # Форма (N, K)
    xi    = np.random.normal(0, sigma, size=(N, K))
    L_ik  = L_mean[:, np.newaxis] + xi             # (N, K)

    # Принятая мощность и ОСШ (N, K)
    P_rx  = received_power_dBm(L_ik)               # дБм
    snr   = snr_linear(P_rx)                       # линейный

    # Максимальная скорость и объём за слот (байт)
    rate  = max_rate(snr)                           # бит/с, (N, K)
    V_ik  = data_volume_slot(rate)                  # байт,  (N, K)

    # Приход пакетов: A_ik ~ Geom(p), количество пакетов
    A_pkt = np.random.geometric(p, size=(N, K))    # (N, K)
    A_ik  = A_pkt * PACKET_SIZE                     # байт,  (N, K)

    # Буфер каждого АБ (байт)
    buf   = np.zeros(N)
    total_buf_per_slot = np.zeros(K)

    for k in range(K):
        # Пополнение буфера
        buf += A_ik[:, k]

        # Round-Robin: в слоте k обслуживается абонент с индексом k mod N
        i_served = k % N
        sent     = min(buf[i_served], V_ik[i_served, k])
        buf[i_served] = max(0.0, buf[i_served] - sent)

        total_buf_per_slot[k] = np.sum(buf)

    # Среднее суммарное значение по всем слотам
    return np.mean(total_buf_per_slot)


# ═══════════════════════════════════════════════════════════════════════════
#  6. ПРИОРИТЕТНЫЙ АЛГОРИТМ (Proportional Fair)
# ═══════════════════════════════════════════════════════════════════════════
def simulate_proportional_fair(N, p, K=K, window=20):
    """
    Алгоритм Proportional Fair (PF).
    В каждом слоте передача осуществляется абоненту с максимальным
    приоритетом: priority_i = C_ik / avg_C_i,
    где avg_C_i — скользящее среднее скорости абонента i.
    """
    d       = place_subscribers(N, R)
    L_mean  = okumura_hata_large_city(d)
    xi      = np.random.normal(0, sigma, size=(N, K))
    L_ik    = L_mean[:, np.newaxis] + xi
    P_rx    = received_power_dBm(L_ik)
    snr     = snr_linear(P_rx)
    rate    = max_rate(snr)
    V_ik    = data_volume_slot(rate)
    A_ik    = np.random.geometric(p, size=(N, K)) * PACKET_SIZE

    buf         = np.zeros(N)
    avg_rate    = np.ones(N) * np.mean(V_ik)   # инициализация
    total_buf_per_slot = np.zeros(K)
    alpha       = 1 / window   # коэффициент экспоненциального сглаживания

    for k in range(K):
        buf += A_ik[:, k]

        # Приоритет: текущая скорость / средняя скорость
        priorities  = V_ik[:, k] / (avg_rate + 1e-9)
        i_served    = int(np.argmax(priorities))

        # Обновление скользящего среднего
        avg_rate = (1 - alpha) * avg_rate
        avg_rate[i_served] += alpha * V_ik[i_served, k]

        sent = min(buf[i_served], V_ik[i_served, k])
        buf[i_served] = max(0.0, buf[i_served] - sent)

        total_buf_per_slot[k] = np.sum(buf)

    return np.mean(total_buf_per_slot)


# ═══════════════════════════════════════════════════════════════════════════
#  7. ЗАПУСК МОДЕЛИРОВАНИЯ
# ═══════════════════════════════════════════════════════════════════════════
np.random.seed(42)   # для воспроизводимости

print("=== Моделирование Round-Robin ===")
rr_results  = {}   # {N: массив средних буферов по p}
pf_results  = {}   # {N: массив средних буферов по p}

for N in N_list:
    rr_buf = []
    pf_buf = []
    for p in p_vals:
        rr = simulate_round_robin(N, p)
        pf = simulate_proportional_fair(N, p)
        rr_buf.append(rr)
        pf_buf.append(pf)
        print(f"  N={N:3d}, p={p:.1f} | RR буфер: {rr/1024:.1f} кБ | PF буфер: {pf/1024:.1f} кБ")
    rr_results[N] = np.array(rr_buf)
    pf_results[N] = np.array(pf_buf)


# ═══════════════════════════════════════════════════════════════════════════
#  8. ГРАФИКИ
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Лаб. работа №2. Модель Окумура-Хата (large city)\n"
    f"R={R} м, PTX={PTX_W} Вт, f0={f0} МГц, kn={kn}",
    fontsize=13
)

colors  = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['o', 's', '^']

# График 1 — Round-Robin
ax1 = axes[0]
for (N, col, mk) in zip(N_list, colors, markers):
    ax1.plot(p_vals, rr_results[N] / 1024, color=col, marker=mk,
             label=f"N = {N}")
ax1.set_xlabel("Параметр p", fontsize=12)
ax1.set_ylabel("Средний суммарный объём данных в буфере, кБ", fontsize=11)
ax1.set_title("Алгоритм Round-Robin", fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xticks(p_vals)

# График 2 — Proportional Fair
ax2 = axes[1]
for (N, col, mk) in zip(N_list, colors, markers):
    ax2.plot(p_vals, pf_results[N] / 1024, color=col, marker=mk,
             label=f"N = {N}")
ax2.set_xlabel("Параметр p", fontsize=12)
ax2.set_ylabel("Средний суммарный объём данных в буфере, кБ", fontsize=11)
ax2.set_title("Алгоритм Proportional Fair", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_xticks(p_vals)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/lab2_graphs.png", dpi=150)
plt.show()
print("Графики сохранены: lab2_graphs.png")
