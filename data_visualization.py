import os
from PIL import Image
import matplotlib.pyplot as plt

# Given data
epochs = list(range(1, 28))
miou_values = [
    0.4492968954791884, 0.5291312852301667, 0.5342273423716318, 0.57943716308401,
    0.561596670864331, 0.572516747074321, 0.5955170200403918, 0.602033940974656,
    0.5956161380877493, 0.5959496106880684, 0.5964885267231309, 0.6155924518202517,
    0.6045913038056321, 0.601777712978622, 0.6020641567910516, 0.6014312489984268,
    0.599912200021573, 0.6016746003606529, 0.5923578068493708, 0.6066392827000007,
    0.6085529509713273, 0.5940965637065998, 0.6015325049800181, 0.6083214748121983,
    0.6061143524057218, 0.598265589643829, 0.601985580799276
]

miou_custom = [0.42955813242145136, 0.521884580794228, 0.5286311542560813, 0.5703189983262857,
               0.5655664305665516, 0.5744965300660454, 0.5926572343425395, 0.5989761004607389, 0.5903024951664998,
               0.5926940565494992, 0.5924438047770182, 0.6115361983301772, 0.6007819892682773, 0.5979589056227818,
               0.5988173765413934, 0.5976027519368573, 0.593886877510533, 0.597549400479067, 0.5882888074523683,
               0.6010085493808637, 0.6058320283425607, 0.59052544052181, 0.5985169240303228, 0.605265726823976,
               0.60148594990344, 0.5946663156220954, 0.5975987882380325]

def plot_mIoU_CE():
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, miou_values, marker='o', linestyle='-', color='b')
    plt.title('mIoU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.show()

def plot_mIoU_Custom():
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, miou_custom, marker='o', linestyle='-', color='b')
    plt.title('mIoU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.show()
'''
2007_000129
2008_000359
2008_003926
2008_007797
2010_000679
'''
def plot_images_side_by_side(image_folder):
    # Get a list of filenames in the image folder
    image_filenames = sorted(os.listdir(image_folder))
    image_filenames.pop()
    num_images = len(image_filenames)

    # Set up the figure
    fig, axs = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))
    fig.subplots_adjust(hspace=0.05)
    cnt = 1
    for i, filename in enumerate(image_filenames):
        # Load RGB image
        image_path = os.path.join(image_folder, filename)
        rgb_image = Image.open(image_path)

        # Load Ground Truth annotation
        gt_path = os.path.join("datasets/Vis/output/", filename)
        gt_annotation = Image.open(gt_path)

        # Load Model 1 prediction
        model1_path = os.path.join('datasets/Vis/model1/', filename[:-4] + ".png")
        model1_prediction = Image.open(model1_path)

        # Load Model 2 prediction
        model2_path = os.path.join('datasets/Vis/model2/', filename[:-4] + ".png")
        model2_prediction = Image.open(model2_path)

        # Plot RGB image
        axs[i, 0].imshow(rgb_image)
        axs[i, 0].axis('off')
        if cnt == 1:
            axs[i, 0].set_title('RGB Image')

        # Plot Ground Truth annotation
        axs[i, 1].imshow(gt_annotation)
        axs[i, 1].axis('off')
        if cnt == 1:
            axs[i, 1].set_title('Ground Truth')

        # Plot Model 1 prediction
        axs[i, 2].imshow(model1_prediction)
        axs[i, 2].axis('off')
        if cnt == 1:
            axs[i, 2].set_title('Model 1 Prediction')

        # Plot Model 2 prediction
        axs[i, 3].imshow(model2_prediction)
        axs[i, 3].axis('off')
        if cnt == 1:
            axs[i, 3].set_title('Model 2 Prediction')

        cnt -= 1

    plt.show()


if __name__ == '__main__':
    #plot_mIoU_CE()
    #plot_mIoU_Custom()

    image_folder = "datasets/Vis/RGBimages"

    plot_images_side_by_side(image_folder)

