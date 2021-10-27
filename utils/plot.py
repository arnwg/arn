import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('legend', fontsize=18)

def plt_Gen(rec_losses, bce_gamma, kl_beta, tmp_an):
   
    num_epochs = len(bce_gamma)
    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), bce_gamma[:num_epochs], 'yellow', label='BCE GAMMA')
    plt.plot(range(1, num_epochs + 1), rec_losses[:num_epochs], 'green', label='MSE')
    plt.plot(range(1, num_epochs + 1), kl_beta[:num_epochs], 'red', label=r'KL BETA')
    
    plt.plot(range(1, num_epochs + 1), tmp_an[:num_epochs], 'magenta', label='Temperature')
    
    plt.legend()
    plt.show()

def plt_Gen_noTemp(rec_losses, bce_gamma, kl_beta):

    num_epochs = len(bce_gamma)
    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), bce_gamma[:num_epochs], 'yellow', label='BCE GAMMA')
    plt.plot(range(1, num_epochs + 1), rec_losses[:num_epochs], 'green', label='MSE')
    plt.plot(range(1, num_epochs + 1), kl_beta[:num_epochs], 'red', label=r'KL BETA')

    plt.legend()
    plt.show()


def plot_ARN_loss(d_losses, g_losses, d_losses_val, bce_losses, rec_losses, kldes, real_scores, fake_scores):
    
    num_epochs = len(d_losses)

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), d_losses[:num_epochs], label=r'$\mathcal{L}_{D}$')
    plt.plot(range(1, num_epochs + 1), g_losses[:num_epochs], label=r'$\mathcal{L}_{G}$')
    plt.plot(range(1, num_epochs + 1), d_losses_val[:num_epochs], '--', label=r'L$_{D_{val}}$')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), bce_losses[:num_epochs], 'yellow', label=r'$ -\mathbb{E}_{\tilde{x}} \left[\log p_{\theta} (y|\tilde{x}) \right]$')
    plt.plot(range(1, num_epochs + 1), rec_losses[:num_epochs], 'green', label=r'$\mathbb{E}_{\tilde{x}}\left[\log p(x|\tilde{x})\right]$')
    plt.plot(range(1, num_epochs + 1), kldes[:num_epochs], 'red', label=r'$\mathbb{KLD}$')
    #plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.ylim(0, 1)
    plt.plot(range(1, num_epochs + 1), real_scores[:num_epochs], label=r'$\mathbb{E}_{x}\left[p_{\theta} (0|x)\right]$')
    plt.plot(range(1, num_epochs + 1), fake_scores[:num_epochs], label=r'$\mathbb{E}_{\tilde{x}}\left[p_{\theta} (0|\tilde{x})\right]$')
    plt.legend()
    plt.show()

def plot_ARN_KLD_loss(d_losses, g_losses, d_losses_val, bce_losses, rec_losses, real_scores, fake_scores):

    num_epochs = len(d_losses)

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), d_losses[:num_epochs], label=r'$\mathcal{L}_{D}$')
    plt.plot(range(1, num_epochs + 1), g_losses[:num_epochs], label=r'$\mathcal{L}_{G}$')
    plt.plot(range(1, num_epochs + 1), d_losses_val[:num_epochs], '--', label=r'L$_{D_{val}}$')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), bce_losses[:num_epochs], 'yellow', label=r'$ -\mathbb{E}_{\tilde{x}} \left[\log p_{\theta} (y|\tilde{x}) \right]$')
    plt.plot(range(1, num_epochs + 1), rec_losses[:num_epochs], 'green', label=r'$\mathbb{E}_{\tilde{x}}\left[\log p(x|\tilde{x})\right]$')
    #plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.ylim(0, 1)
    plt.plot(range(1, num_epochs + 1), real_scores[:num_epochs], label=r'$\mathbb{E}_{x}\left[p_{\theta} (0|x)\right]$')
    plt.plot(range(1, num_epochs + 1), fake_scores[:num_epochs], label=r'$\mathbb{E}_{\tilde{x}}\left[p_{\theta} (0|\tilde{x})\right]$')
    plt.legend()
    plt.show()


def plot_pr_curve(precision, recall):
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def plot_auc_curve(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def plot_ARN_GE_loss(losses, p_true, p_fake, rec_errors, kldes, real_scores, fake_scores):
    num_epochs = len(losses)

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), losses[:num_epochs], label='$\mathcal{L}$')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), p_true[:num_epochs], 'y', label=r'$\mathbb{E}_{x}\left[\log p_{\theta} (1|x)\right]$')
    plt.plot(range(1, num_epochs + 1), p_fake[:num_epochs], 'green', label=r'$\mathbb{E}_{\tilde{x}}\left[\log p(0|\tilde{x})\right]$')
    plt.plot(range(1, num_epochs + 1), rec_errors[:num_epochs], 'magenta', label=r'$\mathbb{E}_{\tilde{x}}\left[\log p_{\theta} (x|\tilde{x})\right]$')
    plt.plot(range(1, num_epochs + 1), kldes[:num_epochs], 'red', label=r'$\mathbb{KLD}$')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.ylim(0, 1)
    plt.plot(range(1, num_epochs + 1), fake_scores[:num_epochs], label='fake score')
    plt.plot(range(1, num_epochs + 1), real_scores[:num_epochs], label='real score')

    plt.legend()
    plt.show()


def plot_ARN_loss_noV(d_losses, g_losses, bce_losses, rec_losses, kldes, real_scores, fake_scores):
    num_epochs = len(d_losses)

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), d_losses[:num_epochs], label=r'$\mathcal{L}_{D}$')
    plt.plot(range(1, num_epochs + 1), g_losses[:num_epochs], label=r'$\mathcal{L}_{G}$')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), bce_losses[:num_epochs], 'yellow', label=r'$ -\mathbb{E}_{\tilde{x}} \left[\log p_{\theta} (y|\tilde{x}) \right]$')
    plt.plot(range(1, num_epochs + 1), rec_losses[:num_epochs], 'green', label=r'$\mathbb{E}_{\tilde{x}}\left[\log p(x|\tilde{x})\right]$')
    plt.plot(range(1, num_epochs + 1), kldes[:num_epochs], 'red', label=r'$\mathbb{KLD}$')
    #plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.ylim(0, 1)
    plt.plot(range(1, num_epochs + 1), real_scores[:num_epochs], label=r'$\mathbb{E}_{x}\left[p_{\theta} (0|x)\right]$')
    plt.plot(range(1, num_epochs + 1), fake_scores[:num_epochs], label=r'$\mathbb{E}_{\tilde{x}}\left[p_{\theta} (0|\tilde{x})\right]$')
    plt.legend()
    plt.show()


