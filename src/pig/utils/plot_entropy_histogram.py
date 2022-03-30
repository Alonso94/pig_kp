import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size']= 5


def plot_entropy(image,entropy):
    # plot the image and the entropy
    # create a subplot
    fig,axes=plt.subplots(1,3)
    # fig.suptitle(text)
    # plot the original image
    axes[0].imshow(image[:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    # axes[0].imshow(image.detach().cpu().permute(1,2,0).numpy().astype(np.uint8), cmap='gray')
    axes[0].set_title('Original image')
    # plot the entropy of the RGB channels
    axes[1].imshow(entropy[0].detach().cpu().numpy(),cmap='jet')
    axes[1].set_title('Entropy of the RGB channels', fontsize=4)
    # # plot the entropy of the depth channels
    axes[2].imshow(entropy[0].detach().cpu().numpy(),cmap='jet')
    axes[2].set_title('Entropy of the depth channels', fontsize=4)
    # remove the axis ticks and the borders
    for ax in axes.flat:
        # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ax.axis('off')
    # plt.tight_layout()
    plt.show()
    
def plot_joint_entropy(image,joint_entropy):
    # plot the image and the entropy
    # create a subplot
    fig,axes=plt.subplots(2,2)
    # plot the original image
    axes[0,0].imshow(image[0,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    axes[0,1].imshow(image[1,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    # axes[0].set_title('Original image')
    # plot the entropy of the RGB channels
    axes[1,0].imshow(joint_entropy[0].detach().cpu().numpy(),cmap='jet')
    axes[1,0].set_title('Joint entropy of the RGB channels', fontsize=4)
    # plot the entropy of the depth channels
    axes[1,1].imshow(joint_entropy[1].detach().cpu().numpy(),cmap='jet')
    axes[1,1].set_title('Joint entropy of the depth channels', fontsize=4)
    # remove the axis ticks and the borders
    for ax in axes.flat:
        # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ax.axis('off')
    # plt.tight_layout()
    plt.show()

def plot_conditional_entropy(image,conditional_entropy, joint_entropy, entropy):
    # plot the image and the entropy
    # create a subplot
    fig,axes=plt.subplots(3,2)
    # plot the original image
    axes[0,0].imshow(image[0,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    axes[0,1].imshow(image[1,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    # axes[0].set_title('Original image')
    # plot the entropy of the first image
    axes[1,0].imshow(entropy[0].detach().cpu().numpy(),cmap='jet')
    axes[1,0].set_title('Entropy of the first image', fontsize=3, pad=-10)
    # plot the entropy of the second image
    axes[1,1].imshow(entropy[1].detach().cpu().numpy(),cmap='jet')
    axes[1,1].set_title('Entropy of the second image', fontsize=3, pad=-10)
    # plot the joint entropy
    axes[2,0].imshow(joint_entropy.detach().cpu().numpy(),cmap='jet')
    axes[2,0].set_title('Joint entropy of the both images', fontsize=3, pad=-10)
    # plot the conditional entropy
    axes[2,1].imshow(conditional_entropy.detach().cpu().numpy(),cmap='jet')
    axes[2,1].set_title('Conditional entropy', fontsize=3,pad=-10)
    # remove the axis ticks and the borders
    for ax in axes.flat:
        # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_histogram(image,histogram):
    # plot the image and the entropy
    # create a subplot
    fig,axes=plt.subplots(1,2)
    # plot the original image
    axes[0].imshow(image[:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    axes[0].axis('off')
    # axes[0].set_title('Original image')
    # plot the histogram of the image
    axes[1].bar(range(256),histogram[0].detach().cpu().numpy(),color='r',alpha=0.3)
    axes[1].bar(range(256),histogram[1].detach().cpu().numpy(),color='b',alpha=0.3)
    axes[1].bar(range(256),histogram[2].detach().cpu().numpy(),color='g',alpha=0.3)
    axes[1].set_title('Histogram of the RGB channels')
    # plt.tight_layout()
    plt.show()