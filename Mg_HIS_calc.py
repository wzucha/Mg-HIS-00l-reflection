# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:45:45 2023

@author: wzucha
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

plt.rc('font', size=10)
plt.rcParams['font.sans-serif'] = "Arial"
WAVELENGTHS = 1.79026 # Co alpha

# a.df.loc[a.df["Atom"] == 'Mg+2', "occ"] = 0.15 # backup

# Coefficients for analytical approximation: https://it.iucr.org/Cb/ch6o1v0001/table6o1o1o4.pdf
COEFS_SCATTERING = {
    #"Na+": [3.25650, 2.66710, 3.93620, 6.11530, 1.39980, 0.200100, 1.00320, 14.0390, 0.404000],
    #"K+": [7.95780, 12.6331, 7.49170, 0.767400, 6.35900, -0.00200, 1.19150, 31.9128, -4.9978],
    #"Rb+": [17.5816, 1.71390, 7.65980, 14.7957, 5.89810, 0.160300, 2.78170, 31.2087, 2.07820],
    #"Cs+": [20.3524, 3.55200, 19.1278, 0.308600, 10.2821, 23.7128, 0.961500, 59.4565, 3.27910],
    #"Li+": [0.696800, 4.62370, 0.788800, 1.95570, 0.341400, 0.631600, 0.156300, 10.0953, 0.016700],
    #"Cl-": [18.2915, 0.006600, 7.20840, 1.17170, 6.53370, 19.5424, 2.33860, 60.4486, -16.378],
    #"Br-": [17.1718, 2.20590, 6.33380, 19.3345, 5.57540, 0.287100, 3.72720, 58.1535, 3.17760],
    #"I-": [20.2332, 4.35790, 18.9970, 0.381500, 7.80690, 29.5259, 2.88680, 84.9304, 4.07140],
    "Si+4": [4.43918, 1.64167, 3.20345, 3.43757, 1.19453, 0.214900, 0.416530, 6.65365, 0.746297],
    "Al+3": [4.17448, 1.93816, 3.38760, 4.14553, 1.20296, 0.228753, 0.528137, 8.28524, 0.706786],
    "Mg+2": [3.49880, 2.16760, 3.83780, 4.75420, 1.32840, 0.18500, 0.849700, 10.1411, 0.485300],
    "O-2": [4.19160, 12.8573, 1.63949, 4.17236, 1.52673, 47.0179, -20.307, -0.01404, 21.9412],
    "iO-2": [4.19160, 12.8573, 1.63949, 4.17236, 1.52673, 47.0179, -20.307, -0.01404, 21.9412],
    "Ca+2": [15.6348, -0.00740, 7.95180, 0.608900, 8.43720, 10.3116, 0.853700, 25.9905, -14.875],
    }

# creats an array from 4 to 60 twotheta with a stepsize of 0.02
TWOTHETA = np.arange(3, 25, 0.01)

################# FUNCTIONS ##########

def scattering_factor(koefs, twotheta, wavelenght):

    summe = 0

    for i in range(0,7,2):
        summe += koefs[i]*np.exp(-koefs[i+1]*
                                 ((np.sin(np.deg2rad(twotheta/2)))**2)
                                 /wavelenght**2)
    summe += koefs[-1]
    return summe



def thermale(thermal_coef, twotheta, wavelenght):
    summe = np.exp(thermal_coef*
                   ((np.sin(np.deg2rad(twotheta/2)))**2)/wavelenght**2)
    return summe

def structurefactor(twotheta, wavelenght, l, structure):

    summe = 0
    temp_cos = 0
    temp_sin = 0

    temp_summe = []

    for idx, row in structure.iterrows():

        atom = COEFS_SCATTERING[row["Atom"]]
        beq = row["beq"]
        occ = row["occ"]
        fc = scattering_factor(atom, twotheta, wavelenght)
        ft = thermale(beq, twotheta, wavelenght)


        temp_c = occ*fc*ft*np.cos(2*np.pi*(l*row["z"]))
        temp_s = occ*fc*ft*np.sin(2*np.pi*(l*row["z"]))

        temp_summe.append([row["Atom"], temp_c, temp_s])

        temp_cos += temp_c
        temp_sin += temp_s

    summe = temp_cos**2 + temp_sin**2

    return summe, temp_summe

def LPfactor(twotheta_arr):
    twotheta = twotheta_arr
    nenner = 1 + np.cos(np.deg2rad(twotheta/2))**2
    zaehler = (np.sin(np.deg2rad(twotheta/2))**2) * (np.cos(np.deg2rad(twotheta/2)))
    
    Lp = nenner/zaehler
    
    return Lp

def gauss(twotheta_arr, i_hkl, g_coef):
    return np.exp((-(twotheta_arr-i_hkl)**2)/g_coef)

def cubic(h,k,l, lattice, wavelenght): #hkl indeces
    
    d_value = lattice/np.sqrt(h**2 + k**2 + l**2)
    twotheta = dvalue2twotheta(d_value, wavelenght)

    return twotheta

def monoclinic(l, c_value, wavelenght):
    """
    CALCULATES ONLY THE D SPACING FOR THE 00L REFLEXES

    Parameters
    ----------
    l : TYPE
        DESCRIPTION.
    lattice : TYPE
        DESCRIPTION.
    wavelenght : TYPE
        DESCRIPTION.

    Returns
    -------
    POSITION (IN 2°THETA OF THE 00L PEAKS.

    """

    d_value  =  c_value/l
    twotheta = dvalue2twotheta(d_value, wavelenght)
    return twotheta



def dvalue2twotheta(dvalue, wavelenght):

    twotheta = 2*np.rad2deg(np.arcsin(np.minimum(1,wavelenght/(2*dvalue))))
    return twotheta



def peak_indices(lattice, wavelenght):

    peaklist = []
    hkl_idx = []
    for l in range(1,4):
        temp = monoclinic(l, lattice, wavelenght)
        peaklist.append(temp)
        hkl_idx.append([l])
    uni, idx, counts = np.unique(peaklist, return_index=True,return_counts=True)

    hkl = []
    for i in idx:
        hkl.append(hkl_idx[i])
    return hkl, uni, counts


def norm(array):
    temp = max(array)
    arr = []
    for F in array:
        arr.append(F/temp)
    return arr

class overview():
    def __init__(self, index, data):
        self.index = index
        self.df = np.array(data)
        self.contribution = []

        self.composition()

    def composition(self):
        temp = np.unique(self.df[:,0])
        for i in temp:
            cont = self.df[self.df[:,0] == i]
            self.contribution.append([cont[0,0], np.sum(cont[:,1].astype(float)),
                                      np.sum(cont[:,2].astype(float))])

        self.contribution = np.array(self.contribution)

class mmt:
    def __init__(self):
        self.c_value = 14.8 # change c_value
        self.wavelenght = 1.79026 # change wavelength here

        self.indices = ["001", "002", "003"]
        _ ,self.peak_position, _ = peak_indices(self.c_value, self.wavelenght)
        self.df = pd.read_excel("mmt.xlsx")

        self.intensities = [] # store the intensities 
        self.overview = []
        #self.calc_intensities()

    def calc_intensities(self):
        for idx, ele in enumerate(self.indices):
            temp, df = structurefactor(self.peak_position[idx],
                                             self.wavelenght,
                                             int(ele), self.df)
            self.overview.append(overview(ele, df))
            LP = LPfactor(self.peak_position[idx])
            temp = temp * LP
            self.intensities.append(temp)
            self.int = norm(self.intensities) #necessairy for unknown reasons

start_time = time.time()

occ = np.arange(0, 1.01, 0.01)
zz = np.arange(-0.05, 0.051, 0.001)


class storage:
    def __init__(self, arr_1, arr_2):
        self.df = np.zeros((len(arr_1), len(arr_2)))


b_001 = storage(occ,zz)
b_002 = storage(occ,zz)
b_003 = storage(occ,zz)


for idx, ele in enumerate(occ):
    for idxx, elee in enumerate(zz):
        a = mmt()
        a.df.loc[a.df["Label"] == 'il', "occ"] = ele
        temp = a.df.loc[a.df["Label"] == "il", "z"] + elee
        a.df.loc[a.df["Label"] == "il", "z"] = temp
        a.calc_intensities()
        x,y,z = a.int
        b_001.df[idx, idxx] = x
        b_002.df[idx, idxx] = y
        b_003.df[idx, idxx] = z
    print(f"now we calc step {idx}")


########## FIGURE #######

cmap = "plasma"

fig, (ax, ax1, ax2) = plt.subplots(1, 3, dpi=400, figsize=(9,3))
df = ax.pcolormesh(zz,occ, b_001.df, vmin=0, vmax=np.max(b_001.df), cmap=cmap)
cf = ax1.pcolormesh(zz,occ, b_002.df, vmin=0, vmax=np.max(b_002.df), cmap=cmap)
cf = ax2.pcolormesh(zz,occ, b_003.df, vmin=0, vmax=np.max(b_001.df), cmap=cmap)

ax1.set_yticklabels([])
ax2.set_yticklabels([])

ax.set_box_aspect(1)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)

ax.set_title("001 reflection", fontsize=10)
ax1.set_title("002 reflection", fontsize=10)
ax2.set_title("003 reflection", fontsize=10)


ax.set_ylabel("Interlayer occupancy [%]")

fig.supxlabel("Change of the atom position in z-direction", y=0.05, fontsize=10)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.25, 0.02, 0.5])
fig.colorbar(cf, cax=cbar_ax)
fig.savefig("intensities.png")

print("--- %s seconds ---" % (time.time() - start_time))