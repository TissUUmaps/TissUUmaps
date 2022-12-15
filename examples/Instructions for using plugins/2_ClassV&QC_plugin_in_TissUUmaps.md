# Using TissUUmaps plugin ClassV&QC

ClassV&QC plugin is for visualization, comparison and quality control of cell classification. This plugin was designed for multiplexed microscopy data (multiple images of the same tissue but with different staining target).

## Classification comparison:

> **Note**
> To simplify saving TissUUmaps projects, always keep all files used in a common project (`.tif` and `.csv`) inside one folder.

1. Open the menu **Plugins -> Add plugin**, check the boxes in front of the plugin ClassV&QC, click Install, and re-start TissUUmaps.
1. Open TissUUmaps.
1. Load all image channels into TissUUmaps. You can do it by dragging and dropping all the individual `.tif` image files into TissUUmaps space. All layers will be listed in the tab Image layers.
1. Load the .csv file containing classification results of the first method (for example, manual annotations) by clicking on tab **Markers**, clicking the `[+]` button, and selecting the desired file from your computer under the section **File and coordinates - Choose file**.

1. You can change the Tab name to any desired name, for example, _Manual annotations_, then you need to select the column names from the `.csv` file corresponding to the X and Y coordinates.
1. In the section **Render Options**, you can define a key to group by, that is a column from the .csv file which will be used to display the dataset grouped by different colors and shapes of the marker.
1. Click the **[Update View]** button.
1. Now repeat steps 3-6 but for the .csv file containing classification results of the second method (for example, neural network classifier).

1. Save the project as a `.tmap` file by opening the menu **File -> Save project**. In order to save the project together with the `.csv` file, it is necessary to generate a button first. A warning window appears and you need to generate the button. The path to the `.csv` file needs to be relative to the path of the image.
1. Then you select a suitable directory to save the project and write the project file name, i.e. `My_project.tmap`, and the project is saved.
1. Open the plugin `ClassV&QC` by clicking the menu **Plugins -> ClassQC**.
1. In the plugin’s dropdown menu `Select marker Dataset 1` - select the first dataset (_Manual annotations_), in the dropdown menu `Select column of Dataset 1` - select the column of Dataset 1 containing the classification results, in the dropdown menu `Select marker Dataset 2` - select the automated classification dataset (_Neural network classifier_), in the dropdown menu `Select column of Dataset 2` - select the column of Dataset 2 containing the classification results.
1. Check the box to see the confusion matrix. The user can click on the elements of the confusion matrix and only cells counted in that matrix element are displayed on the Spatial viewport. This function requires that the two methods that are compared have the same cell segmentation/identification as input so that the order of the cells matches. Non-matching cell IDs only enable visualization of cell type distributions and patches of microscopy data.
1. Click the Display button and explore misclassification from the spatial point of view and use the interactive confusion matrix. You can click on the cells and see the cropped patches from all the channels. The size of the displayed patches can be adjusted in the settings Figure size.

   ![ClassQC](images/ClassQC.gif?raw=true)

> **Note**
> In case something is not clear you can always search for help in our documentation: https://tissuumaps.github.io/TissUUmaps-docs/.
