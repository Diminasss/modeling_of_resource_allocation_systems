import numpy as np
import matplotlib.pyplot as plt

# ПАРАМЕТРЫ СЕТИ (Вариант 2)
R = 10  # радиус соты, м
P_tx = 0.01  # мощность передатчика БС, Вт
f0_MHz = 2400  # несущая частота, МГц
delta_f = 20e6  # ширина полосы канала, Гц
kn = 3  # показатель степени потерь (ITU indoor)
T = 290  # температура, К
k_B = 1.38e-23  # постоянная Больцмана, Дж/К
N_ITER = 100  # число реализаций случайного размещения
N_VALUES = [1, 2, 4, 8, 16, 32, 64, 100]  # число абонентов
D_MIN = 0.1  # минимальное расстояние от БС, м

# ШУМОВАЯ МОЩНОСТЬ
P_noise = k_B * T * delta_f  # Вт


# ФУНКЦИИ
def path_loss_dB(d):
    """
    Потери на трассе по модели ITU (office area).
    d : расстояние от БС, м
    Возвращает L в дБ.
    """
    return 20.0 * np.log10(f0_MHz) + 10.0 * kn * np.log10(d) - 28.0


def max_rates(distances):
    """
    Вычисляет предельно достижимые скорости по формуле Шеннона.
    distances : массив расстояний, м
    Возвращает массив C_i, бит/с.
    """
    L_dB = path_loss_dB(distances)  # потери, дБ
    P_rx = P_tx * 10.0 ** (-L_dB / 10.0)  # принятая мощность, Вт
    SNR = P_rx / P_noise  # линейный SNR
    return delta_f * np.log2(1.0 + SNR)  # бит/с


def place_users(N):
    """
    Равномерно размещает N абонентов внутри круга радиуса R.
    Возвращает (x, y, distances).
    """
    phi = np.random.uniform(0, 2 * np.pi, N)
    r = R * np.sqrt(np.random.uniform(0, 1, N))
    r = np.maximum(r, D_MIN)  # избегаем d = 0
    x, y = r * np.cos(phi), r * np.sin(phi)
    return x, y, r


def scheduler_prs(C):
    """
    ПРС — Планировщик Равных Скоростей.
    Выделяет ресурсы обратно пропорционально C_i.
    """
    alpha = (1.0 / C) / np.sum(1.0 / C)
    return alpha * C  # все r_i одинаковы = 1/sum(1/C_j)


def scheduler_pss(C):
    """
    ПСС — Планировщик максимальной Суммарной Скорости.
    Все ресурсы — абоненту с наибольшим C_i.
    """
    rates = np.zeros_like(C)
    rates[np.argmax(C)] = C[np.argmax(C)]
    return rates


def scheduler_prd(C):
    """
    ПРД — Планировщик Равных Долей ресурсов.
    alpha_i = 1/N для всех.
    """
    return C / len(C)


def compute_metrics(rates):
    """
    Считает суммарную, среднюю и минимальную скорости.
    """
    return np.sum(rates), np.mean(rates), np.min(rates)


def main():
    # ГЛАВНЫЙ ЦИКЛ МОДЕЛИРОВАНИЯ
    # Структура: stats[планировщик][метрика] = список средних по N_VALUES
    SCHEDULERS = {
        'ПРС': scheduler_prs,
        'ПСС': scheduler_pss,
        'ПРД': scheduler_prd,
    }

    stats = {
        name: {'R_sum': [], 'R_avg': [], 'R_min': []}
        for name in SCHEDULERS
    }

    for N in N_VALUES:
        # Временные накопители для усреднения по N_ITER реализациям
        buf = {name: {'R_sum': [], 'R_avg': [], 'R_min': []}
               for name in SCHEDULERS}

        for _ in range(N_ITER):
            _, _, distances = place_users(N)
            C = max_rates(distances)

            for name, sched_fn in SCHEDULERS.items():
                rates = sched_fn(C)
                r_sum, r_avg, r_min = compute_metrics(rates)
                buf[name]['R_sum'].append(r_sum)
                buf[name]['R_avg'].append(r_avg)
                buf[name]['R_min'].append(r_min)

        for name in SCHEDULERS:
            stats[name]['R_sum'].append(np.mean(buf[name]['R_sum']))
            stats[name]['R_avg'].append(np.mean(buf[name]['R_avg']))
            stats[name]['R_min'].append(np.mean(buf[name]['R_min']))

    # ============================================================
    # ПРИМЕР ОТОБРАЖЕНИЯ РАСПОЛОЖЕНИЯ (для отчёта, пункт 2)
    # ============================================================
    fig_place, ax_place = plt.subplots(figsize=(5, 5))
    x_ex, y_ex, d_ex = place_users(100)
    ax_place.scatter(x_ex, y_ex, color='blue', label='Абоненты')
    ax_place.scatter(0, 0, color='red', s=100, marker='^', label='БС')
    circle = plt.Circle((0, 0), R, fill=False, color='gray', linestyle='--')
    ax_place.add_patch(circle)
    ax_place.set_xlim(-R * 1.2, R * 1.2)
    ax_place.set_ylim(-R * 1.2, R * 1.2)
    ax_place.set_aspect('equal')
    ax_place.set_title('Пример размещения 16 абонентов (R=10 м)')
    ax_place.set_xlabel('x, м')
    ax_place.set_ylabel('y, м')
    ax_place.legend()
    ax_place.grid(True)
    plt.tight_layout()
    plt.savefig('placement_example.png', dpi=150)
    plt.show()

    # ГРАФИКИ ЗАВИСИМОСТИ МЕТРИК ОТ N
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics_cfg = [
        ('R_sum', 'Суммарная скорость $R_{\\Sigma}$', 'Мбит/с'),
        ('R_avg', 'Средняя скорость $\\bar{R}$', 'Мбит/с'),
        ('R_min', 'Средняя минимальная скорость $R_{\\min}$', 'Мбит/с'),
    ]

    colors = {'ПРС': 'blue', 'ПСС': 'green', 'ПРД': 'red'}

    for ax, (metric, title, unit) in zip(axes, metrics_cfg):
        for name in SCHEDULERS:
            values_Mbps = [v / 1e6 for v in stats[name][metric]]
            ax.plot(N_VALUES, values_Mbps, marker='o',
                    label=name, color=colors[name])
        ax.set_title(title)
        ax.set_xlabel('Число абонентов N')
        ax.set_ylabel(f'{title.split(" ")[0]}, {unit}')
        ax.set_xscale('log', base=2)
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels(N_VALUES)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.suptitle('Зависимость показателей качества обслуживания от числа абонентов\n'
                 f'(ITU office, R={R} м, Ptx={P_tx} Вт, f₀={f0_MHz} МГц, '
                 f'Δf={int(delta_f / 1e6)} МГц, kn={kn})',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('metrics_vs_N.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ВЫВОД ЧИСЛЕННЫХ РЕЗУЛЬТАТОВ В КОНСОЛЬ
    print(f"\n{'=' * 65}")
    print(f"{'Параметры:'} R={R}м, Ptx={P_tx}Вт, f0={f0_MHz}МГц, "
          f"Δf={int(delta_f / 1e6)}МГц, kn={kn}")
    print(f"Мощность шума: {P_noise:.3e} Вт  ({10 * np.log10(P_noise / 1e-3):.1f} дБм)")
    print(f"{'=' * 65}")
    print(f"{'N':>4} | {'Планировщик':>6} | {'R_сум, Мбит/с':>14} | "
          f"{'R_ср, Мбит/с':>13} | {'R_мин, Мбит/с':>14}")
    print(f"{'-' * 65}")
    for i, N in enumerate(N_VALUES):
        for name in ['ПРС', 'ПСС', 'ПРД']:
            print(f"{N:>4} | {name:>6} | "
                  f"{stats[name]['R_sum'][i] / 1e6:>14.3f} | "
                  f"{stats[name]['R_avg'][i] / 1e6:>13.3f} | "
                  f"{stats[name]['R_min'][i] / 1e6:>14.3f}")
        print(f"{'-' * 65}")


if __name__ == '__main__':
    main()
