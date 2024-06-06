import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def read_coordinates_from_file(file_path, radius_filter):
    coordinates = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():
                parts = line.split('Coordinates: ')
                radius_str = parts[0].split('Radius:')[1].strip().rstrip(',')
                radius = float(radius_str)

                if radius == radius_filter:
                    # Pegar coordenadas sem o parse complexo anterior
                    coord_pairs = parts[1].strip()[1:-2].split('), (')
                    for coord_pair in coord_pairs:
                        x_str, y_str = coord_pair.strip('()').split(',')
                        x = float(x_str.strip())
                        y = float(y_str.strip())
                        coordinates.append((x, y))
    return coordinates

def main():
    st.title('Heatmap')
    file_path = 'coordinates_by_radius_001.txt'
    radius = st.slider("Radius", min_value=0.01, max_value=0.49, value=0.01, step=0.01)

    coords = read_coordinates_from_file(file_path, radius)

    if coords:
        coords_int = np.array([(round(x), round(y)) for x, y in coords], dtype=int)
        coords_array = np.array(coords)
        heatmap, xedges, yedges = np.histogram2d(coords_array[:,0], coords_array[:,1], bins=(50, 50))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
        plt.colorbar()
        plt.title('Heatmap das Coordenadas')
        plt.xlabel('x')
        plt.ylabel('y')
        st.pyplot(plt)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].hist(coords_int[:, 0], bins=20, color='blue')
        axs[0].set_title('')
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('Frequência')
        axs[0].set_xlim(0, 300) 

        axs[1].hist(coords_int[:, 1], bins=20, color='green')
        axs[1].set_title('')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('Frequência')
        axs[1].set_xlim(0, 300)

        st.pyplot(fig)
    else:
        st.write("Nenhuma coordenada disponível para a resolução selecionada.")

if __name__ == "__main__":
    main()
