""" Functions for matching only and creating a csv 
with the new global coords and if they are 
to be erased ----------------------------------Relabel2 """
# nb2
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import IPython.display
import os
from skimage import io

# nb 3
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.morphology import dilation
from skimage.morphology import erosion
from skimage.feature import peak_local_max
import sys

filenamecolumn = "Image_FileName_DNA"
po_x_location = "Nuclei_Location_Center_X"
po_y_location = "Nuclei_Location_Center_Y"
# features is the size of the tile (not all tiles are the same #size for instance in the borders)
tilewidth = "Image_Width_DNA"
tileheight = "Image_Height_DNA"
df_nuclei = None
df_nuclei_mod = None
df_voronoi = None
df_image = None
df_experiment = None


def tableNamesFromDB(dbfilename):
    db = sqlite3.connect(dbfilename)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tablenames = []
    for table_name in tables:
        table_name = table_name[0]
        tablenames.append(str(table_name))

    cursor.close()
    db.close()
    return tablenames


def loadIntoDataFrame(csvfilename):
    df = pd.read_csv(csvfilename)
    return df


def loadDBIntoDataFrame(dbfilename):
    tables = tableNamesFromDB(dbfilename)
    numtables = len(tables)
    df_po = None
    df_so = None
    df_im = None
    for t in range(numtables):
        print("[" + str(t) + "] " + tables[t])
    print("[" + str(numtables + 1) + "] None")
    print("Select table for primary objects")
    db = sqlite3.connect(dbfilename)
    po = input()
    po = int(po)
    if po <= (numtables):
        print("selected " + tables[po])
        df_po = pd.read_sql_query("SELECT * from " + tables[po], db)
        # print(df_po.iloc[0])

    print("Select table for Secondary objects")
    so = input()
    so = int(so)
    if so <= (numtables):
        print("selected " + tables[so])
        df_so = pd.read_sql_query("SELECT * from " + tables[so], db)
        # print(df_so.iloc[0])

    print("Select table for image data")
    idata = input()
    idata = int(idata)
    if idata <= (numtables):
        print("selected " + tables[idata])
        df_im = pd.read_sql_query("SELECT * from " + tables[idata], db)
        # print(df_im.iloc[0])

    print("Select table for experiment data")
    idata = input()
    idata = int(idata)
    if idata <= (numtables):
        print("selected " + tables[idata])
        df_exp = pd.read_sql_query("SELECT * from " + tables[idata], db)
        # print(df_im.iloc[0])

    db.close()
    return df_po, df_so, df_im, df_exp


def drawNeighbors(im, ir, dfm, dfr, neighbor, overlap, grace):
    matchedm = dfm.loc[dfm["Matched"] != -1]
    matchedr = dfr.loc[dfr["Matched"] != -1]

    f = None
    axarr = None
    if neighbor == "right":
        f, axarr = plt.subplots(1, 2, figsize=(7, 4))
    elif neighbor == "bottom":
        f, axarr = plt.subplots(2, 1, figsize=(4, 7))

    axarr[0].imshow(im, cmap="gray")
    axarr[1].imshow(ir, cmap="gray")

    x = matchedm[po_x_location].values
    y = matchedm[po_y_location].values

    axarr[0].plot(dfm[po_x_location].values, dfm[po_y_location].values, "r+")
    axarr[0].plot(x, y, "gx")
    if neighbor == "right":
        axarr[0].axvline(x=im.shape[1] - grace * overlap)
        axarr[0].axvline(x=im.shape[1] - overlap, color="red")
    elif neighbor == "bottom":
        axarr[0].axhline(y=im.shape[0] - grace * overlap)
        axarr[0].axhline(y=im.shape[0] - overlap, color="red")

    count = 0
    for xx, yy in zip(x, y):
        circ = Circle((xx, yy), 18, fill=False, edgecolor="red")
        axarr[0].add_patch(circ)
        axarr[0].annotate(
            str(count),
            (xx, yy),
            color="w",
            weight="bold",
            fontsize=6,
            ha="center",
            va="center",
        )
        count += 1

    x = matchedr[po_x_location].values
    y = matchedr[po_y_location].values

    axarr[1].plot(dfr[po_x_location].values, dfr[po_y_location].values, "r+")
    axarr[1].plot(x, y, "gx")
    if neighbor == "right":
        axarr[1].axvline(x=grace * overlap)
        axarr[1].axvline(x=overlap, color="red")
    elif neighbor == "bottom":
        axarr[1].axhline(y=grace * overlap)
        axarr[1].axhline(y=overlap, color="red")

    count = 0
    for xx, yy in zip(x, y):
        circ = Circle((xx, yy), 20, fill=False, edgecolor="red")
        axarr[1].add_patch(circ)
        axarr[1].annotate(
            str(count),
            (xx, yy),
            color="w",
            weight="bold",
            fontsize=6,
            ha="center",
            va="center",
        )
        count += 1

    plt.figure(figsize=(20, 20))
    plt.show()


def pointDist(p1, p2, xoffset=0, yoffset=0):
    """Gets two rows from dataframes and compares them spatially in L2
    The point has to already be unwrapped since rows come from row[1]
    """
    x1 = p1[po_x_location]
    y1 = p1[po_y_location]
    x2 = p2[po_x_location]
    y2 = p2[po_y_location]
    # print (x1,x2+xoffset)
    # print (y1,y2+yoffset)
    d = np.sqrt(
        (x1 - (x2 + xoffset)) * (x1 - (x2 + xoffset))
        + (y1 - (y2 + yoffset)) * (y1 - (y2 + yoffset))
    )
    return d


def matchDFs(mborder, nborder, xoffset=0, yoffset=0, maxdistance=8):
    for mi, mb in mborder.iterrows():
        if mb["Matched"] == -1:
            for ri, rb in nborder.iterrows():
                # if the point is marked for deletion then there is no need to match it
                if rb["Matched"] == -1:
                    d = pointDist(mb, rb, xoffset=xoffset, yoffset=yoffset)
                    if d < maxdistance:
                        # print mborder.at[mi,"Global_Exp_ID"],rborder.at[ri,"Global_Exp_ID"]
                        mborder.at[mi, "Matched"] = nborder.at[ri, "Global_Exp_ID"]
                        nborder.at[ri, "Matched"] = mborder.at[mi, "Global_Exp_ID"]
                        df_nuclei_mod.at[mi, "Matched"] = nborder.at[
                            ri, "Global_Exp_ID"
                        ]
                        df_nuclei_mod.at[ri, "Matched"] = mborder.at[
                            mi, "Global_Exp_ID"
                        ]
    return mborder, nborder


def goThroughNeighbor(
    mobjects,
    robjects,
    inm,
    inr,
    neighbor,
    tilesize,
    overlap,
    grace=2,
    usesob=False,
    draw=False,
    debug=False,
):
    """
    [m,r]objects are dataframes containing info to match
    in[m,r] image numbers in the experiment of the main image and its neighbor
    neighbor: right or bottom str
    tilesize: in pixels, no overlap
    overlap: overlap...
    grace: multiplier to the overlap since a center can be on the very edge yet we still want to match it
    usesob: use secondary objects, def false since sometimes we dont want them
    draw: print results
    debug: if true prints more info
    """

    # get all the objects for m and for r
    loc = po_x_location
    # features is the size of the tile (not all tiles are the same
    # size for instance in the borders)
    xoff = tilesize
    yoff = 0
    if neighbor == "right":
        loc = po_x_location
        feature = tilewidth
        xoff = tilesize
        yoff = 0
    if neighbor == "bottom":
        loc = po_y_location
        feature = tileheight
        xoff = 0
        yoff = tilesize

    if debug:
        if len(mobjects) == 0:
            print("Warning: len(mobjects) is 0, nothing to match to")
            print(inm, inr)
        elif len(robjects) == 0:
            print("Warning: len(robjects) is 0, nothing to match to")
            print(inm, inr)
    # else:
    #    print("No warning")
    #    print inm,inr

    # If the tile is on the border, its size is different, so just in case, bring the size
    # These are copies not references
    mfeature = df_image.loc[df_image["ImageNumber"] == inm][feature].values[0]
    # rfeature=df_image.loc[df_image["ImageNumber"]==inr][feature].values[0]

    # get from main, all objects at its correct border. These are copies, not references
    mborder = mobjects.loc[mobjects[loc] > mfeature - grace * overlap]
    rborder = robjects.loc[robjects[loc] < grace * overlap]

    if debug:
        print("Before matching")
        IPython.display.display(mborder.loc[:, ["Erase", "Matched", "Global_Exp_ID"]])
        IPython.display.display(rborder.loc[:, ["Erase", "Matched", "Global_Exp_ID"]])

    # matchDFs function assigns the Matched column to the GID of the match
    # if a point has not match it is left with -1
    mborder, rborder = matchDFs(
        mborder, rborder, xoffset=xoff, yoffset=yoff, maxdistance=10
    )

    # Lets look only in the matched ones
    matchedm = mborder.loc[(mborder["Matched"] != -1) & (mborder["Erase"] == False)]
    # matchedr=rborder.loc[(rborder["Matched"]!=-1) & (rborder["Erase"]==False)]

    if debug:
        print("After matching")
        IPython.display.display(mborder.loc[:, ["Erase", "Matched", "Global_Exp_ID"]])
        IPython.display.display(rborder.loc[:, ["Erase", "Matched", "Global_Exp_ID"]])

    count = -1
    printinfo = debug
    # After matches have been found, flag for Deletion marking the Erase column
    # Depending on the conditions below
    for mi, mb in matchedm.iterrows():
        rb = None
        ri = None
        try:
            rb = df_nuclei_mod.loc[df_nuclei_mod["Global_Exp_ID"] == mb["Matched"]]
            ri = rb.index[0]
        except IndexError as e:
            print("+--------------------------------------------------------------+")
            print("+---ERROR!!----------------------------------------------------+")
            print("inm,inr,neighbor")
            print(inm, inr, neighbor)
            print("mi,mb")
            print(mi)
            print(mb)
            print(e)
            print("+--------------------------------------------------------------+")

        mgid = mb["Global_Exp_ID"]
        mcvc = None
        if usesob:
            mcvc = mb["Children_voronoi_Count"]

        rgid = rb["Global_Exp_ID"].values[0]
        rcvc = None
        if usesob:
            rcvc = rb["Children_voronoi_Count"]

        count += 1

        # if one of them has no children, meaning no voronoi,
        # then set the childless to be the same global id as the one with children
        if (mcvc == 0) & (rcvc != 0):
            # mborder.at[mi,"Global_Exp_ID"]=rb["Global_Exp_ID"].values[0]
            mborder.at[mi, "Erase"] = True
            df_nuclei_mod.at[mi, "Erase"] = True
            # Global_Exp_ID ImageNumber ObjectNumber

            if printinfo:
                print(str(count), "(mcvc==0) & (rcvc!=0) erase childless from images")
            continue
        elif (mcvc != 0) & (rcvc == 0):
            # rborder.at[ri,"Global_Exp_ID"]=mb["Global_Exp_ID"]
            rborder.at[ri, "Erase"] = True
            df_nuclei_mod.at[ri, "Erase"] = True
            if printinfo:
                print(str(count), "(mcvc!=0) & (rcvc==0) erase childless from images")
            continue
        elif (mcvc == 0) & (rcvc == 0):
            mborder.at[mi, "Erase"] = True
            df_nuclei_mod.at[mi, "Erase"] = True
            rborder.at[ri, "Erase"] = True
            df_nuclei_mod.at[ri, "Erase"] = True
            # This case is tricky because of their Y location, too up, like in 17-18_6.png
            # If it is not solved in right it might be in bottom , if it is in the pure corner
            # I will ahve to deal with this case
            if printinfo:
                print(str(count), "(mcvc==0) & (rcvc==0) no children")
            continue

        # If none was in the border of the image having 0 children
        mf = mb[loc]
        rf = rb[loc].values[0]
        md = mfeature - mf
        rd = rf

        if md < rd:
            # mborder.at[mi,"Global_Exp_ID"]=rb["Global_Exp_ID"].values[0]
            mborder.at[mi, "Erase"] = True
            df_nuclei_mod.at[mi, "Erase"] = True
            if printinfo:
                print(str(count) + "r is further away from the border in " + neighbor)
            continue
        else:
            # rborder.at[ri,"Global_Exp_ID"]=mb["Global_Exp_ID"]
            rborder.at[ri, "Erase"] = True
            df_nuclei_mod.at[ri, "Erase"] = True
            if printinfo:
                print(
                    str(count)
                    + "md>rd m is further away from the border in "
                    + neighbor
                )
            continue


def matchObjectsInBorders(
    x,
    y,
    mainlocation,
    tilesize=1024,
    overlap=100,
    grace=2.1,
    filetypeorig=".png",
    filetypelabel=".png",
    suffix="labels",
    draw=True,
):
    """Matches all the objects in the borders of tiles"""

    images = {"main": None, "bottom": None, "right": None, "diagonal": None}
    file_nuc_bases = {
        "bottom": str(x) + "_" + str(y + 1),
        "right": str(x + 1) + "_" + str(y),
    }
    ImageNumbers = {"main": None, "bottom": None, "right": None, "diagonal": None}
    hasNeighbor = {"bottom": False, "right": False}

    m = str(x) + "_" + str(y) + suffix + filetypelabel
    path = mainlocation + m
    if os.path.isfile(path):
        images["main"] = io.imread(path)
        # get the current image number from the center tile in this experiment given by x an y
        ImageNumbers["main"] = df_image.loc[
            df_image[filenamecolumn] == str(x) + "_" + str(y) + filetypeorig
        ]["ImageNumber"].values[0]
    else:
        # This is teh case when the tile x y doesnt exist
        return False

    for n in hasNeighbor:
        path = mainlocation + file_nuc_bases[n] + suffix + filetypelabel
        if os.path.isfile(path):
            images[n] = io.imread(path)
            # get the current image number from the neighbor tiles in this experiment given by x+ an y+
            vals = df_image.loc[
                df_image[filenamecolumn] == file_nuc_bases[n] + filetypeorig
            ]["ImageNumber"].values
            if vals.size > 0:
                # print (vals)
                ImageNumbers[n] = vals[0]
                # ImageNumbers[n]=df_image.loc[df_image[filenamecolumn]==file_nuc_bases[n]+filetypeorig]["ImageNumber"].values[0]
            else:
                print("Values was 0, break at:", x, y, ImageNumbers[n])

            mobjects = df_nuclei_mod.loc[
                df_nuclei_mod["ImageNumber"] == ImageNumbers["main"]
            ]
            bobjects = df_nuclei_mod.loc[
                df_nuclei_mod["ImageNumber"] == ImageNumbers[n]
            ]

            # obejcts in the main image
            for index, row in mobjects.iterrows():
                ##if global coords are done dont change them again.
                # if (df_nuclei_mod.at[index,"Global_X"]<0):
                df_nuclei_mod.at[index, "Global_X"] = (
                    row[po_x_location] + x * tilesize - overlap
                )
                # if (df_nuclei_mod.at[index,"Global_Y"]<0):
                df_nuclei_mod.at[index, "Global_Y"] = (
                    row[po_y_location] + y * tilesize - overlap
                )

            # objects in neighbor image
            for index, row in bobjects.iterrows():
                ##if global coords are done dont change them again.
                # if (df_nuclei_mod.at[index,"Global_X"]<0):
                df_nuclei_mod.at[index, "Global_X"] = (
                    row[po_x_location] + x * tilesize - overlap
                )
                # if (df_nuclei_mod.at[index,"Global_Y"]<0):
                df_nuclei_mod.at[index, "Global_Y"] = (
                    row[po_y_location] + y * tilesize - overlap
                )

            goThroughNeighbor(
                mobjects,
                bobjects,
                ImageNumbers["main"],
                ImageNumbers[n],
                n,
                tilesize,
                overlap,
                grace=grace,
                draw=draw,
            )

            if draw:
                mobjects = df_nuclei_mod.loc[
                    df_nuclei_mod["ImageNumber"] == ImageNumbers["main"]
                ]
                bobjects = df_nuclei_mod.loc[
                    df_nuclei_mod["ImageNumber"] == ImageNumbers[n]
                ]
                drawNeighbors(
                    images["main"], images[n], mobjects, bobjects, n, overlap, grace
                )
            hasNeighbor[n] = True


""" Functions for matching only and creating a csv 
with the new global coords and if they are 
to be erased ----------------------------------Relabel3 """


def appendNeighbors(alist, x, y):
    alist.append((x - 1, y - 1))
    alist.append((x, y - 1))
    alist.append((x - 1, y))

    alist.append((x + 1, y + 1))
    alist.append((x, y + 1))
    alist.append((x + 1, y))

    alist.append((x - 1, y + 1))
    alist.append((x - 1, y + 1))


def localFloodFill(seed, value, replace, image):
    """Do flood fill only on the current component
    Args:
        seed (tuple): x,y tuple.
        value (int): value to look for.
        replace (int): replace with this value, def 0
        image (ndarray): 2D image to look into
    """
    if value == 0:
        # print ("finished flodfill value is 0")
        return
    neighborList = []
    neighborList.append(seed)
    while len(neighborList) > 0:
        point = neighborList.pop()
        if image[int(point[1]), int(point[0])] == value:
            image[int(point[1]), int(point[0])] = replace
            appendNeighbors(neighborList, int(point[0]), int(point[1]))


def eraseSelectedLabels(
    x,
    y,
    nl,
    vl,
    savelocation,
    filetypeorig=".png",
    filetypelabel=".png",
    nuc_sfx="labels",
    vor_sfx="voronoi",
    draw=False,
    save=False,
):
    filename = str(x) + "_" + str(y) + filetypeorig
    # get the ImageNumber
    inum = df_image.loc[df_image[filenamecolumn] == filename]["ImageNumber"].values[0]
    numplots = 2
    vor_image = None
    # get all nuclei belonging to inum
    nuclei = df_nuclei_mod.loc[
        (df_nuclei_mod["ImageNumber"] == inum) & (df_nuclei_mod["Erase"] == True)
    ]
    # &
    # (df_nuclei_mod[po_x_location].notna()) &
    # (df_nuclei_mod[po_y_location].notna())

    # NaN coordinates... dunno why but they exist
    nucleiNaN = df_nuclei_mod.loc[
        (df_nuclei_mod[po_x_location].isna()) | (df_nuclei_mod[po_y_location].isna())
    ]

    if len(nucleiNaN):
        print("image " + filename + "has NaN coordinates")
        print(nucleiNaN["Global_Exp_ID"])

    # if(draw):
    #    IPython.display.display(nuclei[0:5,:])
    print("Erasing " + str(len(nuclei)) + " elements")

    if len(nuclei) > 0:
        # let's erase some labels, load the images
        # image=None
        # try:
        nuc_image = io.imread(nl + str(x) + "_" + str(y) + nuc_sfx + filetypelabel)
        if vl:
            numplots = 2
            vor_image = io.imread(vl + str(x) + "_" + str(y) + vor_sfx + filetypelabel)
        if draw:
            f, axarr = plt.subplots(1, numplots, figsize=(10, 5))
            axarr[0].imshow(nuc_image, cmap="nipy_spectral")
            if vl:
                axarr[1].imshow(vor_image, cmap="nipy_spectral")
            plt.title("Before")
            plt.show()
        # except:
        #    print( "Could not read image. Return false and continue (x,y,inum,filename)")
        #    print (x,y,inum,nl+str(x)+"_"+str(y)+nuc_sfx+filetypelabel)
        #    if vl:
        #        print("vor image, "+vl+str(x)+"_"+str(y)+vor_sfx+filetypelabel)
        #    print (sys.exc_info()[0])
        #    return False

        arrx = []
        arry = []
        for ni, nuc in nuclei.iterrows():
            # if(np.isnan(nuc[po_x_location]) and np.isnan(nuc[po_y_location]) ):
            # if(nuc[po_x_location].isna() and nuc[po_y_location].isna() ):

            nx = int(np.floor(nuc[po_x_location]))
            ny = int(np.floor(nuc[po_y_location]))
            arrx.append(nx)
            arry.append(ny)
            nucev = nuc_image[ny, nx]
            vorev = None
            localFloodFill((nx, ny), nucev, 0, nuc_image)
            if vl:
                vorev = vor_image[ny, nx]
                localFloodFill((nx, ny), vorev, 0, vor_image)
        if draw:
            f, axarr = plt.subplots(1, numplots, figsize=(10, 5))
            axarr[0].imshow(nuc_image, cmap="nipy_spectral")
            axarr[0].plot(arrx, arry, "gx")
            if vl:
                axarr[1].imshow(vor_image, cmap="nipy_spectral")
                axarr[1].plot(arrx, arry, "gx")
            plt.title("After")
            plt.show()

        if save:
            io.imsave(
                savelocation
                + "labels/"
                + str(x)
                + "_"
                + str(y)
                + "corrected"
                + filetypelabel,
                nuc_image,
            )
            if vl:
                io.imsave(
                    savelocation
                    + "voronoi/"
                    + str(x)
                    + "_"
                    + str(y)
                    + "corrected"
                    + filetypelabel,
                    vor_image,
                )

        return True
    else:
        # still save the iamge as corrected
        if save:
            try:
                nuc_image = io.imread(
                    nl + str(x) + "_" + str(y) + nuc_sfx + filetypelabel
                )
                vor_image = None
                if vl:
                    vor_image = io.imread(
                        vl + str(x) + "_" + str(y) + vor_sfx + filetypelabel
                    )
                if draw:
                    if vl:
                        numplots = 2
                    f, axarr = plt.subplots(1, numplots, figsize=(10, 5))

                    axarr[0].imshow(nuc_image, cmap="nipy_spectral")
                    if vl:
                        axarr[1].imshow(vor_image, cmap="nipy_spectral")
                    plt.title("No change")
                    plt.show()
            except:
                print(
                    "Could not read image. Return false and continue (x,y,inum,filename)"
                )
                print(
                    x,
                    y,
                    inum,
                    nl + str(x) + "_" + str(y) + nuc_sfx + filetypelabel,
                    vl + str(x) + "_" + str(y) + vor_sfx + filetypelabel,
                )
                print(sys.exc_info()[0])
                return False
            io.imsave(
                savelocation
                + "labels/"
                + str(x)
                + "_"
                + str(y)
                + "corrected"
                + filetypelabel,
                nuc_image,
            )
            if vl:
                io.imsave(
                    savelocation
                    + "voronoi/"
                    + str(x)
                    + "_"
                    + str(y)
                    + "corrected"
                    + filetypelabel,
                    vor_image,
                )


def outlineImage(labelimage, cellsize):
    regionsize = cellsize  # parameter that must match CP's
    # at identify secondary objects: Number of pixels by which
    # to expand the primary obejcts, in my case 15
    distance = ndi.distance_transform_edt(labelimage == 0)
    disttocalc = distance <= regionsize
    newl = watershed(disttocalc, labelimage, mask=disttocalc > 0)
    border = ((dilation(newl) - erosion(newl)) > 0) * 255
    return border


def unpadfolder(
    tilesize,
    overlap,
    mainlocation,
    savelocation,
    suffix,
    filetypeorig=".png",
    filetypelabel=".png",
    debug=False,
):
    i = 0
    j = 0
    name = str(i) + "_" + str(j) + suffix + filetypeorig
    hasrightn = 1
    hasbottomn = 1
    while hasrightn or hasbottomn:
        name = str(i) + "_" + str(j) + suffix + filetypeorig
        path = mainlocation + name
        if not os.path.isfile(path):
            print("file ", name, "does not exist")
            i = 0
            j = 0
        print("loading", i, j)
        # file exists, now is it a corner? does it have neighbors to the right and or bottom?
        rn = mainlocation + str(i + 1) + "_" + str(j) + suffix + filetypeorig
        bn = mainlocation + str(i) + "_" + str(j + 1) + suffix + filetypeorig
        tn = mainlocation + str(i) + "_" + str(j - 1) + suffix + filetypeorig
        ln = mainlocation + str(i - 1) + "_" + str(j) + suffix + filetypeorig

        hasrightn = int(os.path.isfile(rn))
        hasbottomn = int(os.path.isfile(bn))
        hastopn = int(os.path.isfile(tn))
        hasleftn = int(os.path.isfile(ln))

        image = io.imread(path)

        border = ((dilation(image) - erosion(image)) > 0) * 255
        border = border.astype("uint8")

        topy = hastopn * overlap
        topx = hasleftn * overlap
        shapey, shapex = image.shape
        shapex -= hasrightn * overlap
        shapey -= hasbottomn * overlap

        if debug:
            print(image.shape)
            print(rn)
            print(bn)
            print(tn)
            print(ln)
            print("hasrightn, hasbottomn, hastopn, hasleftn")
            print(hasrightn, hasbottomn, hastopn, hasleftn)
            print("topx,topy,shapex,shapey,y,x")
            print(topx, topy, shapex, shapey, image.shape[1], image.shape[0])

        if shapex >= image.shape[1]:
            shapex = image.shape[1]

        if shapey >= image.shape[0]:
            shapey = image.shape[0]

        savethis = border[topy:shapey, topx:shapex]

        if debug:
            print(shapex, shapey, image.shape[1], image.shape[0])
            plt.figure(figsize=(10, 10))
            print(savethis.shape)
            plt.imshow(border)
            plt.axvline(x=topx)
            if shapey == image.shape[0]:
                plt.axvline(x=shapex - 1)
            plt.axhline(y=topy)
            if shapex == image.shape[1] - 1:
                plt.axhline(y=shapey)
            plt.show()
            plt.figure(figsize=(10, 10))
            plt.imshow(savethis)
            plt.show()

        print("saving", i, j)
        io.imsave(
            savelocation + str(i) + "_" + str(j) + "border" + filetypelabel, savethis
        )

        if hasrightn:
            i += 1
        else:
            i = 0
            j += 1
        if not hasrightn and not hasbottomn:
            # this is the end, my only friend, the end
            # finished succesfully
            break
