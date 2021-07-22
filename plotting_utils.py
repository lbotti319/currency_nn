import matplotlib.pyplot as plt


def multiple_ysets(arg):
    if isinstance(arg, list):
        if not isinstance(arg[0], (int, float)):
            return True
    return False


def plot_multi_scale(
    xlabel,
    x,
    y_left,
    ylabel_left,
    c_left="b",
    legend_left=None,
    log_left=False,
    y_right=None,
    ylabel_right=None,
    c_right="orange",
    legend_right=None,
    log_right=False,
):
    plots = []
    fig, ax1 = plt.subplots()
    if log_left:
        ax1.set_yscale("log")
    ax1.set_xlabel(xlabel)
    if not multiple_ysets(y_left):
        ax1.set_ylabel(ylabel_left, color=c_left)
        plots.extend(ax1.plot(x, y_left, c=c_left, label=legend_left))
    else:
        ax1.set_ylabel(ylabel_left, color=c_left[0])
        for i, (xc, yc, c) in enumerate(zip(x, y_left, c_left)):
            label = None if legend_left is None else legend_left[i]
            plots.extend(ax1.plot(xc, yc, color=c, label=label))
    ax1.tick_params(axis="y")

    if y_right is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        if log_left:
            ax2.set_yscale("log")
        if not multiple_ysets(y_right):
            ax2.set_ylabel(ylabel_right, color=c_right)
            plots.extend(ax2.plot(x, y_right, c=c_right, label=legend_right))
        else:
            ax2.set_ylabel(ylabel_right, color=c_right[0])
            for i, (xc, yc, c) in enumerate(zip(x, y_right, c_right)):
                label = None if legend_right is None else legend_right[i]
                plots.extend(ax2.plot(xc, yc, color=c, label=label))
        ax2.tick_params(axis="y")

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if legend_left is not None:
        labels = [p.get_label() for p in plots]
        ax1.legend(plots, labels, loc="center right")
    plt.show()


def plot_train_test(train, test):
    x, yl, yr = [], [], []
    for i, history in enumerate([train, test]):
        x.append(range(len(history)))
        yl.append([step[0] for step in history])
        yr.append([step[1] for step in history])

    cl = ["b", "g"]
    cr = ["m", "r"]
    legl = ["train cost", "test cost"]
    legr = ["train acc", "test acc"]
    plot_multi_scale(
        "epochs",
        x,
        y_left=yl,
        ylabel_left="cost",
        c_left=cl,
        legend_left=legl,
        log_left=False,
        y_right=yr,
        ylabel_right="accuracy",
        c_right=cr,
        legend_right=legr,
        log_right=False,
    )


def plot_train_test_regression(train, test):
    y_train = [step[0] for step in train]
    y_test = [step[0] for step in test]
    x = list(range(len(train)))
    plt.plot(x, y_train, label="train_cost")
    plt.plot(x, y_test, label="test_cost")
    plt.legend()
    plt.show()
