import numpy as np
import scipy.interpolate as spline
import scipy.ndimage as nima
from cc import default_config as df_conf
import nibabel as nib
import streamlit as st
import matplotlib.pyplot as plt
from skimage import morphology as nima
from matplotlib.colors import ListedColormap
import seaborn as sns
import os


"""
Shape signature profile Module
"""


def sign_extract(seg, resols, smoothness, points, foot): #Function for shape signature extraction
    splines = get_spline(seg,smoothness, foot)

    sign_vect = np.array([]).reshape(0,points) #Initializing temporal signature vector
    for resol in resols:
        sign_vect = np.vstack((sign_vect, get_profile(splines, n_samples=points, radius=resol)))

    return sign_vect

def sign_fit(sig_ref, sig_fit, points): #Function for signature fitting
    dif_curv = []
    for shift in range(points):
        dif_curv.append(np.abs(np.sum((sig_ref - np.roll(sig_fit[0],shift))**2)))
    return np.apply_along_axis(np.roll, 1, sig_fit, np.argmin(dif_curv))

def compute_angles(pivot, anterior, posterior):
    max_angle = np.pi*2

    def angles(vectors):
        return np.arctan2(vectors[1], vectors[0])
    ap, pp = anterior-pivot, posterior-pivot
    ang_post, ang_ant = angles(pp), angles(ap)
    ang = ang_post - ang_ant

    dif_prof = np.abs(ang - np.roll(ang,1)) > np.pi
    ind_start = np.where(dif_prof)[0][::2]
    ind_end = np.where(dif_prof)[0][1::2]
    zeros = np.zeros_like(ang)
    for in1, in2 in zip(ind_start,ind_end):
        if (ang[in1] - np.roll(ang,1)[in1]) > np.pi:
            zeros[in1:in2] = -2*np.pi
        else:
            zeros[in1:in2] = 2*np.pi
    return (ang + zeros) *180/(np.pi)

def get_profile(tck, n_samples, radius):
    def eval_spline(tck, t):
        y, x = spline.splev(t,tck)
        return np.vstack((y,x))

    t_pivot = np.linspace(0,1, n_samples, endpoint=False)
    pivot = eval_spline(tck, t_pivot)
    t_anterior = np.mod(t_pivot+(1-radius), 1)
    anterior = eval_spline(tck, t_anterior)
    t_posterior = np.mod(t_pivot+radius, 1)
    posterior = eval_spline(tck, t_posterior)

    return compute_angles(pivot, anterior, posterior)

def get_seq_graph(edge):

    dy, dx = np.array([-1,0,1,1,1,0,-1,-1]), np.array([-1,-1,-1,0,1,1,1,0])
    def get_neighbors(node):
        Y, X = node[0]+dy, node[1]+dx
        neighbors = edge[Y, X]
        Y, X = Y[neighbors], X[neighbors]
        return list(zip(Y,X))
    graph = {}
    Y, X = edge.nonzero()
    for node in zip(Y,X):
        graph[node] = get_neighbors(node)
    seq = []
    first_el = (Y[0], X[0])
    seq.append(first_el)
    ext_el = first_el
    act_el = graph[ext_el][0]
    while (first_el != ext_el) or (len(seq)==1):
        ind_el = np.where(np.array(graph[(ext_el)])!=act_el)
        ind_el_uq = np.unique(ind_el[0])

        if len(ind_el_uq)==1:
            ind_el = ind_el_uq[0]
        else:
            acum_dist = []
            for ind in ind_el_uq:
                dist_ = (graph[ext_el][ind][0]-ext_el[0])**2+(graph[ext_el][ind][1]-ext_el[1])**2
                acum_dist.append(dist_)
            min_dist = acum_dist.index(min(acum_dist))
            ind_el = ind_el_uq[min_dist]

        act_el = ext_el
        ext_el = graph[(act_el)][ind_el]
        seq.append(ext_el)
    lst1, lst2 = zip(*seq)

    return (np.array(lst1), np.array(lst2))

def get_spline(seg,s,foot):
    nz = np.nonzero(seg)
    x1,x2,y1,y2 = np.amin(nz[0]),np.amax(nz[0]),np.amin(nz[1]),np.amax(nz[1])
    M0 = seg[x1-5:x2+5,y1-5:y2+5]
    nescala = [4*M0.shape[-2],4*M0.shape[-1]]
    M0 = resizedti(M0,nescala).astype('bool')
    M0_ero = nima.binary_erosion(M0, footprint=foot).astype(M0.dtype)
    con_M0 = np.logical_xor(M0_ero,M0)
    seq = get_seq_graph(con_M0)
    tck, _ = spline.splprep(seq, k=5, s=s)

    return tck, M0, M0_ero, con_M0

def resizedti(img,shape):

    y,x = np.indices(shape)
    x = x/(shape[1]/img.shape[-1])
    y = y/(shape[0]/img.shape[-2])
    return img[y.astype('int'),x.astype('int')]

def find_slice_with_mask(img_mask):
        for i in range(img_mask.shape[0]):
            if np.any(img_mask[i]):
                return i
        return None

def main():

    structure_options = {
        "Default": None,
        "Cruz": np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]]),
        "Retângulo": nima.rectangle(3, 5),  
        "Quadrado": nima.square(3), 
        "Círculo": nima.disk(1), 
        "Diamante": nima.diamond(1)  
    }

    # Lista de nomes das opções para o selectbox
    structure_names = list(structure_options.keys())

    # Selecionando a estrutura a partir do selectbox
    selected_structure_name = st.sidebar.selectbox('Selecione a estrutura:', structure_names)

    # Recuperando a estrutura correspondente
    foot = structure_options[selected_structure_name]

    patients = ['000153', '000155', '000158', '000159', '000160', '000161', '000166', 
                '000168', '000169', '000173', '000175', '000176', '000177', '000178', 
                '000179', '000033', '000034', '000723', '001781']
    selected_patient = st.sidebar.selectbox('Selecione o Paciente:', patients)

    radius = st.sidebar.slider('Resolução para perfil', min_value=0.01, max_value=0.49, value=0.49, step=0.01)
    smooth = st.sidebar.slider('Smooth', min_value=100, max_value=1000, value=700, step=100)
    shift = st.sidebar.slider('Shift', min_value=-250, max_value=250, value=0, step=10) 
    
    
    directory_path = f"data/{selected_patient}"
    file_list = os.listdir(directory_path)
    nii_files = [file for file in file_list if file.endswith('.nii.gz') and file != 'FA_inCCsightspace.nii.gz']
    all_coordinates = []

    for file_name in nii_files:
        full_path = os.path.join(directory_path, file_name)
        img_mask = nib.load(full_path).get_fdata()
        msp = find_slice_with_mask(img_mask)

        if msp is not None:
            img_mask_msp_slice = img_mask[msp]
            tck, M0, M0_ero, con_M0 = get_spline(img_mask_msp_slice, smooth, foot)
            angles = get_profile(tck, n_samples=500, radius=radius)
            min_angle_index = np.argmin(angles)
            t_pivot = np.linspace(0, 1, 500, endpoint=False)
            x, y = spline.splev(t_pivot, tck)
            all_coordinates.append((x[min_angle_index], y[min_angle_index], radius))

            col1, col2, col3 = st.columns(3)  # Criando uma terceira coluna

            with col1:
                fig, ax = plt.subplots(figsize=(7, 7))
                custom_cmap = ListedColormap(['white', 'black'])
                ax.imshow(con_M0, cmap=custom_cmap)
                ax.plot(y, x, linestyle='-', color='green', linewidth=2)
                ax.plot(y[min_angle_index], x[min_angle_index], 'bo', label='Menor ângulo', markersize=6, color='red')
                ax.legend()
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(angles, 'b-')
                ax.plot(min_angle_index, angles[min_angle_index], 'ro', label='Menor ângulo')
                ax.set_xlabel("Ponto pivô")
                ax.set_ylabel("Curvatura")
                ax.legend()
                st.pyplot(fig)

        with col3:  # Ajustando o gráfico com o shift periódico
                        fig, ax = plt.subplots(figsize=(5, 3))
                        rolled_angles = np.roll(angles, shift)
                        ax.plot(rolled_angles, 'b-')
                        adjusted_min_angle_index = (min_angle_index + shift) % 500  # Ajusta o índice do menor ângulo
                        ax.plot(adjusted_min_angle_index, rolled_angles[adjusted_min_angle_index], 'ro', label='Menor ângulo')
                        ax.set_xlabel("Ponto pivô")
                        ax.set_ylabel("Curvatura")
                        ax.legend()
                        st.pyplot(fig)

    if all_coordinates:
        coords_int = np.array([(round(x), round(y)) for x, y, radius in all_coordinates], dtype=int)
        coords_array = np.array(all_coordinates)
        heatmap, xedges, yedges = np.histogram2d(coords_array[:,0], coords_array[:,1], bins=(50, 50))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure(figsize=(10, 2))
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
        plt.colorbar()
        plt.title("Heatmap de Localização de Menores Ângulos")
        plt.xlabel("x")
        plt.ylabel("y")
        st.pyplot(plt)

        # Histogramas para coordenadas X e Y
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].hist(coords_int[:, 0], bins=5, color='blue')
        axs[0].set_title('')
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('Frequência')
        axs[0].set_xlim(0, 300) 

        axs[1].hist(coords_int[:, 1], bins=5, color='green')
        axs[1].set_title('')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('Frequência')
        axs[1].set_xlim(0, 300)

        st.pyplot(fig)

if __name__ == "__main__":
    main()