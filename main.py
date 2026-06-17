from pathlib import Path

from optimizers.optimizers import *
from functions import *
import scipy.optimize as opt
import csv
import os, glob
import matplotlib.pyplot as plt

opts = [
    descent_cnst,
    descent_steepest,
    descent_armijo,
    descent_wolfe
]

funcs = [
    f,
    g,
    rosenbrock,
    ackley,
    himmelblau
]

steepest_linear_opt = opt.minimize_scalar


def prepare(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    [f.unlink() for f in Path(output_dir).iterdir() if f.is_file()]


def task1p2(output_dir, fs, steps, MAX_N_ITER):
    prepare(output_dir)
    eps = 1e-8
    starting_pos = np.array([2.0, 2.0])
    for f in fs:
        res = []
        it_counts = []
        for step in steps:
            optimizer = descent_cnst(f.f, f.grad_f, eps, starting_pos.copy(), step, max_iter=MAX_N_ITER)
            xmin, fmin = optimizer.find_minimum()
            res.append((step, optimizer.iteration_count, xmin, fmin))
            it_counts.append(optimizer.iteration_count)
            print(f'{optimizer}_{f} eps {eps} xmin {xmin} fmin {fmin}')

        filename_base = f'task1_Step_{f}'
        filename_csv = filename_base + '_table.csv'
        filename_plt = filename_base + '_plot.png'
        with open(os.path.join(output_dir, filename_csv), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'N_Iter', 'X_min_x', 'X_min_y', 'F_min'])
            for item in res:
                x_min_x, x_min_y = item[2][0], item[2][1]
                writer.writerow([item[0], item[1], x_min_x, x_min_y, item[3]])
            file.close()

        plt.plot(steps, it_counts, label=f'Iteration count', linewidth=2, linestyle='--', color="red", marker='o')
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Step')
        plt.ylabel('Iterations')
        plt.title(f'Iteration count for pair function {f}')
        plt.grid(True, which='both')

        file_path = os.path.join(output_dir, filename_plt)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()


def task1(output_dir):
    task1p2(output_dir, funcs[:2], [10 ** (-i) for i in range(1, 7)], 1_000_000)


def task2(output_dir):
    task1p2(output_dir, funcs[2:], [10 ** (-i) for i in range(1, 4)], 100_000)


def build_trajectory_plot(output_dir, optimizer, f, string=""):
    x_range = (-6, 6)
    y_range = (-6, 6)
    optimizer.do_history = True
    optimizer.find_minimum()
    trajectory = np.array([item[0] for item in optimizer.history])
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]

    x_grid = np.linspace(x_range[0], x_range[1], 200)
    y_grid = np.linspace(y_range[0], y_range[1], 200)

    x_coords = np.nan_to_num(x_coords, nan=np.nan, posinf=np.nan, neginf=np.nan)
    y_coords = np.nan_to_num(y_coords, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f.f(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(8, 8))

    contours = plt.contour(X, Y, Z, levels=50, cmap='viridis', linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

    plt.plot(x_coords, y_coords, '-o', color='red', markersize=3, label='Траектория ГС')

    plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Старт')
    if not np.isnan(x_coords[-1]) and not np.isnan(y_coords[-1]):
        plt.plot(x_coords[-1], y_coords[-1], 'rX', markersize=10, label='Финиш')

    plt.xlim(x_range)
    plt.ylim(y_range)

    plt.title(f"Траектория для пары [{optimizer}-{f}]")
    filename_plt = f'plot_[{optimizer}-{f}]_{string}_Trajectory.png'
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    file_path = os.path.join(output_dir, filename_plt)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    return optimizer.iteration_count, optimizer.function_call_count, optimizer.grad_call_count


def task31(output_dir):
    fs = funcs[:2]
    steps = [1e-3, 1e-3]
    starting_pos = np.array([2.0, 2.5])
    for i in range(2):
        optimizer = descent_cnst(fs[i].f, fs[i].grad_f, 1e-8, starting_pos.copy(), steps[i], max_iter=10_000,
                                 do_history=True)
        build_trajectory_plot(output_dir, optimizer, fs[i])


def task32(output_dir):
    fs = funcs[2:]
    steps = [10 ** (-i) for i in range(1, 4)]
    starting_pos = np.array([2.0, 2.5])
    for f in fs:
        for step in steps:
            optimizer = descent_cnst(f.f, f.grad_f, 1e-8, starting_pos.copy(), step, max_iter=10_000, do_history=True)
            build_trajectory_plot(output_dir, optimizer, f)


def task3(output_dir):
    prepare(output_dir)
    task31(output_dir)
    task32(output_dir)


def task46table(output_dir, res, filename_csv):
    with open(os.path.join(output_dir, filename_csv), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epsilon', 'N_Iter', 'F_Call_Count', 'Grad_Call_Count', 'X_min_x', 'X_min_y', 'F_min'])
        for item in res:
            writer.writerow(list(item))
        file.close()


def task46plots(output_dir, epsilons, it_counts, function_call_counts, grad_call_counts, filename_plt, pairname):
    plt.plot(epsilons, it_counts, label='Iterations', linewidth=2, linestyle='--', color="red", marker='o')
    plt.plot(epsilons, function_call_counts, label='Function calls', linewidth=2, linestyle='--', color="blue",
             marker='s')
    plt.plot(epsilons, grad_call_counts, label='Grad calls', linewidth=2, linestyle='--', color="green", marker='^')
    plt.legend()
    plt.xscale('log')
    plt.xlabel("Epsilon")
    plt.title(f'Call counts for {pairname}')
    plt.grid(True, which='both')
    file_path = os.path.join(output_dir, filename_plt)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def task4(output_dir):
    prepare(output_dir)
    fs = funcs[:2]
    epsilons = [10 ** (-i) for i in range(1, 9)]
    MAX_N_ITER = 10_000
    starting_pos = np.array([2.0, 2.5])
    for opt in opts[2:]:
        temp = opt(None, None, None, None)
        for f in fs:
            res = []
            it_counts = []
            function_call_counts = []
            grad_call_counts = []
            pairname = f'[{temp}-{f}]'
            for eps in epsilons:
                optimizer = opt(f.f, f.grad_f, eps, starting_pos.copy(), max_iter=MAX_N_ITER)
                xmin, fmin = optimizer.find_minimum()
                res.append((eps, optimizer.iteration_count, optimizer.function_call_count, optimizer.grad_call_count, xmin[0], xmin[1], fmin))
                it_counts.append(optimizer.iteration_count)
                function_call_counts.append(optimizer.function_call_count)
                grad_call_counts.append(optimizer.grad_call_count)
                print(f'{pairname} eps {eps} xmin {xmin} fmin {fmin}')
            filename_base = f'task4_[{temp}-{f}]'
            filename_csv = filename_base + '_table.csv'
            filename_plt = filename_base + '_plot.png'
            task46table(output_dir, res, filename_csv)
            task46plots(output_dir, epsilons, it_counts, function_call_counts, grad_call_counts, filename_plt, pairname)
            optimizer = opt(f.f, f.grad_f, 1e-5, starting_pos.copy(), max_iter=MAX_N_ITER)
            build_trajectory_plot(output_dir, optimizer, f)



def task56table(output_dir, res, filename_csv):
    with open(os.path.join(output_dir, filename_csv), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['P_x', 'P_y', 'N_Iter', 'F_Call_Count', 'Grad_Call_Count'])
        for item in res:
            writer.writerow(list(item))
        file.close()


starting_points = [
    np.array([2.0, 2.5]),
    np.array([3.0, 3.0]),
    np.array([4.0, 1.0]),
    np.array([0.0, 2.0]),
    np.array([-1.0, 3.0]),
]


def task5(output_dir):
    prepare(output_dir)
    fs = funcs[2:]
    for opt in opts[2:]:
        temp = opt(None, None, None, None)
        for f in fs:
            res = []
            for starting_pos in starting_points:
                optimizer = opt(f.f, f.grad_f, 1e-8, starting_pos.copy(), max_iter=10_000)
                res.append((starting_pos[0], starting_pos[1], *build_trajectory_plot(output_dir, optimizer, f,
                                                                                     f'Point({starting_pos[0]}, {starting_pos[1]})')))
            filename_csv = f'task5_[{temp}-{f}]_table.csv'
            task56table(output_dir, res, filename_csv)


def task61(output_dir):
    epsilons = [10 ** (-i) for i in range(1, 9)]
    starting_pos = np.array([2.0, 2.5])
    fs = funcs[:2]
    for f in fs:
        res = []
        it_counts = []
        function_call_counts = []
        grad_call_counts = []
        for eps in epsilons:
            opt = descent_steepest(f.f, f.grad_f, eps, starting_pos.copy())
            xmin, fmin = opt.find_minimum()
            it_counts.append(opt.iteration_count)
            function_call_counts.append(opt.function_call_count)
            grad_call_counts.append(opt.grad_call_count)
            res.append((eps, opt.iteration_count, opt.function_call_count, opt.grad_call_count, xmin[0], xmin[1], fmin))
        filename_csv = f'task6_[Steepest-{f}]_table.csv'
        filename_plt = f'task6_[Steepest-{f}]_table.png'
        pairname = f'[Steepest-{f}]'
        task46table(output_dir, res, filename_csv)
        task46plots(output_dir, epsilons, it_counts, function_call_counts, grad_call_counts, filename_plt, pairname)


def task62(output_dir):
    fs = funcs[2:]
    for f in fs:
        res = []
        for starting_pos in starting_points:
            optimizer = descent_steepest(f.f, f.grad_f, 1e-8, starting_pos.copy(), max_iter=10_000)
            res.append((starting_pos[0], starting_pos[1], *build_trajectory_plot(output_dir, optimizer, f,
                                                                                 f'Point({starting_pos[0]}, {starting_pos[1]})')))
        filename_csv = f'task6_[Steepest-{f}]_table.csv'
        task56table(output_dir, res, filename_csv)


def task6(output_dir):
    prepare(output_dir)
    task61(output_dir)
    task62(output_dir)


def main():
    # task1("task1")
    # task2("task2")
    # task3("task3")
    # task4("task4")
    # task5("task5")
    task6("task6")


if __name__ == "__main__":
    main()
