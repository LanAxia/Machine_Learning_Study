import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cat_acc(epoch_history, update_history, batch_per_epoch, title):
    # 创建子图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)

    acc_max = np.max(epoch_history)
    i_max = np.argmax(epoch_history)
    acc_min = np.min(update_history)
    i_min = np.argmin(update_history) / batch_per_epoch

    epoch_range = np.arange(1, len(epoch_history) + 1)
    update_range = np.arange(1, len(update_history) + 1)
    xticks_range = np.arange(0, len(epoch_history) + 0.1, 5)
    yticks_range = np.arange((acc_min // 0.05 * 0.05), 1.01, 0.1)
    yticks_labels = ['{:.0f}%'.format(x * 100) for x in yticks_range]

    ax.plot(epoch_range, epoch_history, c='#4B7BE5', alpha=0.5, label='History by Epoch', lw=1.5)
    ax.plot(update_range / batch_per_epoch, update_history, c='#A85CF9', alpha=0.5, \
    label='History by Update', lw=1.5)
    ax.scatter([i_min, i_max + 1], [acc_min, acc_max], c='#FFA1A1', alpha=0.5)
    ax.text(i_min + 1 + 0.03 * (xticks_range[-1] - xticks_range[0]), \
        acc_min, 'Min: {:.02f}%'.format(acc_min * 100), color='#F24A72', alpha=0.5)
    ax.text(i_max + 1 - 0.07 * (xticks_range[-1] - xticks_range[0]), \
        acc_max - 0.05 * (yticks_range[-1] - yticks_range[0]), 'Max: {:.02f}%'.\
            format(acc_max * 100), color='#F24A72', alpha=0.5)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks(xticks_range, labels=[int(x) for x in xticks_range], fontsize=8)
    ax.set_yticks(yticks_range, labels=yticks_labels, fontsize=8)
    ax.set_xlabel('Epoch  ({} batches/epoch)'.format(batch_per_epoch), color='k', fontsize=14, \
        fontfamily='Ubuntu Mono derivative Powerline')
    ax.set_ylabel('Category Accuracy', color='k', fontsize=14, \
        fontfamily='Ubuntu Mono derivative Powerline')
    ax.set_title(title, color='k', fontsize=14, \
        fontfamily='Ubuntu Mono derivative Powerline')
    ax.grid(ls='--', axis='both', alpha=0.3, c='#417D7A', lw=0.5)
    ax.legend()
    return fig, ax