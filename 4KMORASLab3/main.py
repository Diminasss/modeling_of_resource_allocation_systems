"""
Лабораторная работа №3
Стратегии распределения ресурсных блоков
в централизованной сети со случайным трафиком

Вариант 2: ITU office area
R=10 м | PTX=0.01 Вт | f0=1200 МГц | kn=3 | NRB=50
"""

import numpy as np
import matplotlib.pyplot as plt
import os

#  Параметры варианта
R = 10  # Радиус соты, м
P_TX_W = 0.01  # Мощность передатчика БС, Вт
F0_MHZ = 1200  # Несущая частота, МГц
K_N = 3  # Количество этажей (модель ITU)
N_RB = 50  # Количество ресурсных блоков в кадре

# Параметры модели распространения ITU-R P.1238 (офис)
N_ITU = 33  # Коэффициент потерь на расстоянии (office, 900-2100 МГц)
SIGMA_DB = 8.0  # СКО замирания, дБ

# Параметры радиоинтерфейса
B_RB_HZ = 200_000.0  # Полоса одного ресурсного блока, Гц (200 кГц по методичке)
T_SLOT_S = 1e-3  # Длительность слота, с (1 мс по методичке)
SPS = int(round(1.0 / T_SLOT_S))  # Слотов в секунду = 1000

#  Тепловой шум
K_B = 1.38e-23  # Постоянная Больцмана, Дж/К
T_KELVIN = 290  # Температура, К
N_FLOOR_W = K_B * T_KELVIN * B_RB_HZ  # Мощность шума на один RB, Вт

#  Параметры симуляции
N_SLOTS = 10_000  # Число временных слотов (~10 с)
PKT_BITS = 8 * 1024  # Размер пакета = 1 кБайт = 8192 бит
LAMBDA_RANGE = np.arange(0.5, 6.5, 0.5)  # λ, пакетов/слот на каждого абонента
USER_COUNTS = [5, 10, 20]  # Рассматриваемые числа абонентов
SEED = 42  # Начальное зерно ГСЧ

OUTPUT_DIR = "outputs/"


# МОДЕЛЬ РАСПРОСТРАНЕНИЯ И КАНАЛ
def itu_mean_loss(d: np.ndarray) -> np.ndarray:
    """
    Среднее затухание по модели ITU-R P.1238 (indoor, office area).

    L_mean_i = 20*log10(f0) + N_itu*log10(d_i) + Lf(kn) - 28  [дБ]
    Lf(kn)   = 15 + 4*(kn - 1)  [дБ]

    d < 1 м принимается за 1 м (защита от log10(0)).
    """
    d = np.maximum(np.asarray(d, dtype=float), 1.0)
    Lf = 15.0 + 4.0 * (K_N - 1.0)
    return 20.0 * np.log10(F0_MHZ) + N_ITU * np.log10(d) + Lf - 28.0


def place_users(n_users: int, rng: np.random.Generator) -> np.ndarray:
    """
    Равномерное случайное размещение абонентов внутри круга радиуса R.

    r = R*sqrt(U), phi = 2*pi*V,  U,V ~ Uniform(0,1).
    sqrt(U) обеспечивает равномерную плотность по площади.

    Возвращает расстояния от БС, м, форма (n_users,).
    """
    u = rng.uniform(0.0, 1.0, n_users)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_users)
    r = R * np.sqrt(u)
    return np.hypot(r * np.cos(phi), r * np.sin(phi))


def build_channel_matrix(distances: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Матрица затуханий канала для всех абонентов, RB и слотов.

    h_ijk = L_mean_i + xi_ijk,   xi_ijk ~ N(0, SIGMA_DB^2)

    Возвращает h_db, форма (n_users, N_RB, N_SLOTS), [дБ].
    """
    L_mean = itu_mean_loss(distances)
    xi = rng.normal(0.0, SIGMA_DB, (len(distances), N_RB, N_SLOTS))
    return L_mean[:, None, None] + xi


def capacity_bits_per_slot(h_db: np.ndarray) -> np.ndarray:
    """
    Максимальная скорость передачи данных через один RB за один слот.

    SINR_ijk = P_TX / (10^(h_ijk/10) * N_floor)
    C_ijk    = B_RB * log2(1 + SINR_ijk) * T_slot   [бит/слот]
    """
    sinr = P_TX_W / (10.0 ** (h_db / 10.0) * N_FLOOR_W)
    return B_RB_HZ * np.log2(1.0 + sinr) * T_SLOT_S


# АЛГОРИТМЫ ПЛАНИРОВАНИЯ
def sched_round_robin(cap_k: np.ndarray, _buf: np.ndarray,
                      _avg: np.ndarray, slot: int) -> np.ndarray:
    """
    Round Robin - циклическое равномерное распределение RB.

    alloc(rb, k) = (k * N_RB + rb) mod N_users

    Возвращает массив назначений (N_RB,): элемент = индекс абонента.
    """
    n_users = cap_k.shape[0]
    return (slot * N_RB + np.arange(N_RB)) % n_users


def sched_max_rate(cap_k: np.ndarray, _buf: np.ndarray,
                   _avg: np.ndarray, _slot: int) -> np.ndarray:
    """
    Max Rate - каждый RB отдаётся абоненту с максимальной мгновенной скоростью.

    alloc(rb) = argmax_i  C_{i,rb,k}
    """
    return np.argmax(cap_k, axis=0)


def sched_proportional_fair(cap_k: np.ndarray, _buf: np.ndarray,
                            avg_rates: np.ndarray, _slot: int) -> np.ndarray:
    """
    Proportional Fair - каждый RB отдаётся абоненту с максимальным
    отношением мгновенной скорости к средней скорости за последнюю секунду.

    alloc(rb) = argmax_i  C_{i,rb,k} / R_avg_i(k)
    """
    prio = cap_k / (avg_rates[:, None] + 1e-10)
    return np.argmax(prio, axis=0)


SCHEDULERS: dict = {
    "Round Robin": sched_round_robin,
    "Max Rate": sched_max_rate,
    "Proportional Fair": sched_proportional_fair,
}


# ОСНОВНАЯ СИМУЛЯЦИЯ
def simulate_one(n_users: int, lam: float, sched_fn, seed: int) -> float:
    """
    Симуляция одного сценария за N_SLOTS слотов.

    Для каждого слота k:
      1. Буфер пополняется: Buf_i += Poisson(lambda) * S_pkt
      2. Планировщик назначает RB абонентам
      3. Tx_i = min(sum_rb C_{i,rb,k}, Buf_i)
      4. Buf_i(k+1) = max(0, Buf_i - Tx_i)
      5. Обновляется скользящее окно средней скорости (1 с = SPS слотов)

    Возвращает среднее суммарное содержимое буферов по всем слотам, бит.
    """
    rng = np.random.default_rng(seed)

    distances = place_users(n_users, rng)
    h_db = build_channel_matrix(distances, rng)  # (n_users, N_RB, N_SLOTS)
    cap = capacity_bits_per_slot(h_db)  # (n_users, N_RB, N_SLOTS)
    arrivals = rng.poisson(lam, (n_users, N_SLOTS))  # (n_users, N_SLOTS)

    buffers = np.zeros(n_users)
    avg_rates = np.ones(n_users) * 1e3
    rate_win = np.zeros((n_users, SPS))
    buf_history = np.empty(N_SLOTS)

    for k in range(N_SLOTS):
        # 1. Приход пакетов
        buffers += arrivals[:, k] * PKT_BITS

        # 2. Планирование
        cap_k = cap[:, :, k]
        allocation = sched_fn(cap_k, buffers, avg_rates, k)

        # 3. Подсчёт переданных бит (сумма ёмкостей назначенных RB)
        tx = np.zeros(n_users)
        np.add.at(tx, allocation, cap_k[allocation, np.arange(N_RB)])

        # 4. Нельзя передать больше, чем есть в буфере
        tx = np.minimum(tx, buffers)
        buffers = np.maximum(buffers - tx, 0.0)

        # 5. Обновление скользящего окна средней скорости
        rate_win[:, k % SPS] = tx
        avg_rates = np.maximum(rate_win.mean(axis=1), 1e-10)

        buf_history[k] = buffers.sum()

    return float(buf_history.mean())


def run_all_experiments() -> dict:
    """
    Запуск симуляции для всех комбинаций: алгоритм x N_users x lambda.
    Возвращает results[sched_name][n_users] = ndarray средних буферов по lambda.
    """
    results = {name: {} for name in SCHEDULERS}
    total = len(SCHEDULERS) * len(USER_COUNTS) * len(LAMBDA_RANGE)
    done = 0

    for name, fn in SCHEDULERS.items():
        for n in USER_COUNTS:
            arr = np.empty(len(LAMBDA_RANGE))
            for idx, lam in enumerate(LAMBDA_RANGE):
                arr[idx] = simulate_one(n, lam, fn, SEED)
                done += 1
                print(f"  [{done:3d}/{total}] {name:20s} | N={n:2d} | "
                      f"lambda={lam:.1f} -> B_avg={arr[idx] / 1e6:.3f} Mbit", flush=True)
            results[name][n] = arr

    return results


# ПОСТРОЕНИЕ ГРАФИКОВ

_U_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
_U_MARKERS = ["o", "s", "^"]
_U_LS = ["-", "--", "-."]

_S_COLORS = {"Round Robin": "#1f77b4", "Max Rate": "#d62728",
             "Proportional Fair": "#2ca02c"}
_S_MARKERS = {"Round Robin": "o", "Max Rate": "s", "Proportional Fair": "^"}


def _suptitle(prefix: str) -> str:
    return (f"{prefix}\nITU office area: R={R} м | P_TX={P_TX_W} Вт | "
            f"f0={F0_MHZ} МГц | kn={K_N} | N_RB={N_RB}")


def plot_per_algorithm(results: dict) -> None:
    """
    Три subplot-а (по одному на алгоритм).
    Кривые = разное число абонентов. Ось X = lambda, ось Y = буфер, Мбит.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(_suptitle("Среднее суммарное содержимое буфера vs lambda (по алгоритмам)"),
                 fontsize=10)

    for ax, (name, data) in zip(axes, results.items()):
        for i, n in enumerate(USER_COUNTS):
            ax.plot(LAMBDA_RANGE, data[n] / 1e6,
                    color=_U_COLORS[i], marker=_U_MARKERS[i],
                    linestyle=_U_LS[i], lw=2, ms=6, label=f"N = {n}")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("lambda, пакетов/слот/абонент")
        ax.set_ylabel("Средний суммарный буфер, Мбит")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot1_per_algorithm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранён: {path}")


def plot_per_users(results: dict) -> None:
    """
    Три subplot-а (по одному на число абонентов).
    Кривые = разные алгоритмы. Ось X = lambda, ось Y = буфер, Мбит.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(_suptitle("Среднее суммарное содержимое буфера vs lambda (сравнение алгоритмов)"),
                 fontsize=10)

    for ax, n in zip(axes, USER_COUNTS):
        for name in SCHEDULERS:
            ax.plot(LAMBDA_RANGE, results[name][n] / 1e6,
                    color=_S_COLORS[name], marker=_S_MARKERS[name],
                    lw=2, ms=6, label=name)
        ax.set_title(f"N = {n} абонентов", fontsize=12, fontweight="bold")
        ax.set_xlabel("lambda, пакетов/слот/абонент")
        ax.set_ylabel("Средний суммарный буфер, Мбит")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot2_per_users.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранён: {path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Лабораторная работа №3 — Стратегии распределения RB")
    print(f"Вариант 2: ITU office | R={R}м | P_TX={P_TX_W}Вт | "
          f"f0={F0_MHZ}МГц | kn={K_N} | N_RB={N_RB}")
    print(f"N_SLOTS={N_SLOTS} | T_slot={T_SLOT_S * 1e3:.1f}мс | "
          f"B_RB={B_RB_HZ / 1e3:.0f}кГц | lambda={LAMBDA_RANGE[0]}..{LAMBDA_RANGE[-1]}\n")

    results = run_all_experiments()

    print("\nПостроение графиков...")
    plot_per_algorithm(results)
    plot_per_users(results)

    print(f"\nГотово. Файлы сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
